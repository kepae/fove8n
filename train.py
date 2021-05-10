# Stateless training method (i.e. doesn't depend on globals).
def train(train_loader, net, optimizer, loss_fn, epochs):
    print('Beginning training.')
    running_loss = 0.0
    for epoch in range(epochs):
        for batch_idx, (inputs, labels) in enumerate(train_loader):

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, batch_idx + 1, running_loss / 2000))


if __name__ == '__main__':
    import argparse
    from data_loader import Fove8nDataset
    from modules import OurNet
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description='fove8n training script')
    # Data Loader parameters.
    parser.add_argument('--root_videos_dir', required=True)
    parser.add_argument('--preprocessed_tensors_dir_pattern', required=True)
    parser.add_argument('--labels_scale', type=float, required=True)
    # Training parameters.
    parser.add_argument('--epochs', type=int, required=True)

    args = parser.parse_args()

    # Load the data.
    train_dataset = Fove8nDataset(root_videos_dir = args.root_videos_dir,
                                  preprocessed_tensors_dir_pattern = args.preprocessed_tensors_dir_pattern,
                                  labels_scale = args.labels_scale)

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # TODO: Don't hardcode these parameters. It works because that's the only data shape we have at the moment.
    # Network.
    net = OurNet(num_channels=64, gru_hidden_size=3, frame_height=224, frame_width=224)

    # Select Mean Square Entropy / L2 loss.
    mse_loss = nn.MSELoss()

    # Optimizer.
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train(train_dataloader, net, optimizer, mse_loss, args.epochs)