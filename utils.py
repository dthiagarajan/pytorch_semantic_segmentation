import matplotlib.pyplot as plt

import torch

from tqdm import trange, tqdm


def display_segmentation(image, target, ax=None):
    if ax:
        ax.imshow(image, cmap='gray')
    else:
        plt.imshow(image, cmap='gray')
    if ax:
        ax.imshow(target, cmap='jet', alpha=0.5)
    else:
        plt.imshow(target, cmap='jet', alpha=0.5)


def set_lr(optimizer, model):
    if model.epoch % 10 == 0:
        new_lr = (1e-2) * (0.97 ** (model.epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        print("Learning rate changed to: %.5f" % new_lr)


def train(model, dataloader, criterion, optimizer, epochs,
          scheduler=None, verbose=False):
    model.train()
    losses = []
    epoch_iter = trange(epochs, position=0) if verbose else range(epochs)
    for epoch in epoch_iter:
        set_lr(optimizer, model)
        tr_loss = 0.
        dl_iter = tqdm(dataloader, position=1,
                       leave=False) if verbose else dataloader
        for x, y in dl_iter:
            optimizer.zero_grad()
            if torch.cuda.is_available():
                x, y = x.cuda(), y.squeeze(dim=1).cuda()
            pred = model(x)
            loss = criterion(pred, y)
            tr_loss += loss.data.cpu()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
        overall_loss = tr_loss / len(dataloader)
        losses.append(overall_loss)
        if verbose:
            print(overall_loss)
        model.epoch += 1
    return losses

def display_predictions(model, dataloader, total=5, model_name="Model"):
    count = 0
    for i, (x, y) in enumerate(dataloader):
        if torch.cuda.is_available():
            x, y = x.cuda(), y.squeeze(dim=1).cuda()
        output = torch.argmax(model(x), dim=1)
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
        f.suptitle('ISBI EM Segmentation (%s Prediction vs. GT)' % model_name, fontsize=16)
        display_segmentation(x[0, 0, ...].cpu().numpy(),
                            output.squeeze(dim=0).cpu().numpy(),
                            ax=ax1)
        ax1.set_title("%s Prediction" % model_name)
        display_segmentation(x[0, 0, ...].cpu().numpy(),
                            y.squeeze(dim=0).cpu().numpy(),
                            ax=ax2)
        ax2.set_title("GT")
        count += 1
        if count == 5:
            break
