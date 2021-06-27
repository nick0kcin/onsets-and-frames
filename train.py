import os
from datetime import datetime
from pathlib import Path

import numpy as np
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluate import evaluate
from onsets_and_frames import *
from onsets_and_frames.dataset import MelFramesDataset

ex = Experiment('train_transcriber')


@ex.config
def config():
    # logdir = 'runs/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S')
    logdir = 'runs/transcriber-210610-200355'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 4
    iterations = 2000000 // batch_size
    resume_iteration = 180000
    checkpoint_interval = 10000
    train_on = 'CUSTOM'

    # batch_size = 8
    sequence_length = 327680
    model_complexity = 32

    # if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
    #     batch_size //= 2
    #     sequence_length //= 2
    #     print(f'Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory')

    learning_rate = 0.0006
    learning_rate_decay_steps = 10000
    learning_rate_decay_rate = 0.98

    leave_one_out = None

    clip_gradient_norm = 3

    validation_length = 1672
    validation_interval = 50000 // batch_size

    ex.observers.append(FileStorageObserver.create(logdir))


@ex.automain
def train(logdir, device, iterations, resume_iteration, checkpoint_interval, train_on, batch_size, sequence_length,
          model_complexity, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, leave_one_out,
          clip_gradient_norm, validation_length, validation_interval):
    # print_config(ex.current_run)

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    train_groups, validation_groups = ['train'], ['validation']

    if leave_one_out is not None:
        all_years = {'2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017'}
        train_groups = list(all_years - {str(leave_one_out)})
        validation_groups = [str(leave_one_out)]

    if train_on == 'MAESTRO':
        dataset = MAESTRO(groups=train_groups, sequence_length=sequence_length)
        validation_dataset = MAESTRO(groups=validation_groups, sequence_length=sequence_length)
    elif train_on == "CUSTOM":
        dataset = MelFramesDataset(Path("/home/nick/PycharmProjects/MusicCrawler/train"), iterations - resume_iteration)
        validation_dataset = MelFramesDataset(Path("/home/nick/PycharmProjects/MusicCrawler/val"))

    else:
        dataset = MAPS(groups=['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'], sequence_length=sequence_length)
        validation_dataset = MAPS(groups=['ENSTDkAm', 'ENSTDkCl'], sequence_length=validation_length)

    loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True, num_workers=1)
    val_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

    if resume_iteration is None:
        model = OnsetsAndFrames(N_MELS, MAX_MIDI - MIN_MIDI + 1, model_complexity, use_mel=train_on != "CUSTOM").to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model-{resume_iteration}.pt')
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        print("loaded")

    # summary(model)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    with tqdm(range(resume_iteration + 1, iterations + 1)) as loop:
        for i, batch in zip(loop, cycle(loader)):
            for key in batch:
                batch[key] = batch[key].to(DEFAULT_DEVICE)
            predictions, losses = model.run_on_batch(batch)

            loss = sum(losses.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if clip_gradient_norm:
                clip_grad_norm_(model.parameters(), clip_gradient_norm)

            for key, value in {'loss': loss, **losses}.items():
                writer.add_scalar(key, value.item(), global_step=i)
                loop.set_postfix_str(f"loss: {loss}")

            if i % validation_interval == 0:
                model.eval()
                with torch.no_grad():
                    for key, value in evaluate(val_loader, model).items():
                        writer.add_scalar('validation/' + key.replace(' ', '_'), np.mean(value), global_step=i)
                model.train()

            if i % checkpoint_interval == 0:
                torch.save(model, os.path.join(logdir, f'model-{i}.pt'))
                torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
