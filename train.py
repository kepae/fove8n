def train(train_loader, net, optimizer, loss_fn, epochs):
    for epoch in epochs:
        for batch_idx, (inputs, labels) in enumerate(train_loader):

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()