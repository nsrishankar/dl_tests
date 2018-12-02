def findLR(model, optimizer, criterion, trainloader, final_value=10, init_value=1e-8):

    model.train() # setup model for training configuration

    num = len(trainloader) - 1 # total number of batches
    mult = (final_value / init_value) ** (1/num)

    losses = []
    lrs = []
    best_loss = 0.
    avg_loss = 0.
    beta = 0.98 # the value for smooth losses
    lr = init_value

    for batch_num, (inputs, targets) in enumerate(trainloader):


        optimizer.param_groups[0]['lr'] = lr

        batch_num += 1 # for non zero value
        inputs, targets = inputs.to(device), targets.to(device) # convert to cuda for GPU usage
        optimizer.zero_grad() # clear gradients
        outputs = model(inputs) # forward pass
        loss = criterion(outputs, targets) # compute loss

        #Compute the smoothed loss to create a clean graph
        avg_loss = beta * avg_loss + (1-beta) *loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)

        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss

        # append loss and learning rates for plotting
        lrs.append(math.log10(lr))
        losses.append(smoothed_loss)

        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            break

        # backprop for next step
        loss.backward()
        optimizer.step()

        # update learning rate
        lr = mult*lr

    plt.xlabel('Learning Rates')
    plt.ylabel('Losses')
    plt.plot(lrs,losses)
    plt.show()
