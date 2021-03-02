def run_train(data_loader,device):

    network.train() #set network to train mode

    epoch_loss = 0
    epoch_psnr = 0
    #for batch in train_loader:
    for iteration, batch in enumerate(data_loader):

        lowres = batch[0].to(device)
        hires = batch[1].to(device)

        preds = network(lowres)
        loss = F.mse_loss(preds,hires)#,reduction='mean')
        epoch_loss += loss.item()

        psnr = 20 * log10(1 / sqrt(loss.item())) 
        epoch_psnr += psnr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = epoch_loss/len(data_loader)
    avg_psnr = epoch_psnr/len(data_loader)

    # for sett in RunBuilder.get_runs(plot_settings):
    #     plot_images = sett.plot_images
    #     if plot_images == True:
    #         rand_plot.plot(rgb_plot = sett.rgb_plot, r=sett.r,figsize = sett.figsize)

    return avg_loss, avg_psnr

def run_valid(data_loader,device):

    with torch.no_grad(): #disable gradient calcs in valid

        network.eval() #set network to eval mode

        epoch_loss = 0
        epoch_psnr = 0
        #for batch in train_loader:
        for iteration, batch in enumerate(data_loader):

            lowres = batch[0].to(device)
            hires = batch[1].to(device)

            preds = network(lowres)
            loss = F.mse_loss(preds,hires)#,reduction='mean')
            epoch_loss += loss.item()

            psnr = 20 * log10(1 / sqrt(loss.item())) 
            epoch_psnr += psnr

        avg_loss = epoch_loss/len(data_loader)
        avg_psnr = epoch_psnr/len(data_loader)

        # for sett in RunBuilder.get_runs(plot_settings):
        #     plot_images = sett.plot_images
        #     if plot_images == True:
        #         rand_plot.plot(rgb_plot = sett.rgb_plot, r=sett.r,figsize = sett.figsize)

        return avg_loss, avg_psnr

if __name__ == '__main__':

    for run in RunBuilder.get_runs(params):

        device = torch.device(run.device) #either cpu or cuda, see params above
        torch.cuda.empty_cache() #empty gpu each time or risk get memory overload

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=run.batch_size) #start a dataloader for training set
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=run.batch_size) #start a dataloader for training set

        network = Network().to(device).train() #Network passed to gpu and put into train mode
        optimizer = optim.SGD(network.parameters(), lr = run.lr) #SGD optimizer for loss function

        comment = f'-{run} <--- Validation Test'
        tb = SummaryWriter(comment=comment)
        #tb.add_graph(network)

        print(run)

        for epoch in range(run.ep_range):

            train_loss, train_psnr = run_train(train_loader,run.device)
            valid_loss, valid_psnr = run_valid(valid_loader,run.device)

            tb.add_scalar('Avg_Train_Loss', train_loss, epoch)
            tb.add_scalar('Avg_Train_PSNR', train_psnr, epoch)
            tb.add_scalar('Avg_Valid_Loss', valid_loss, epoch)
            tb.add_scalar('Avg_Valid_PSNR', valid_psnr, epoch)

            print("===> Epoch {}, Avg_Train_Loss: {:.4f}, Avg_Train_PSNR: {:.4f}"
                        .format(epoch, train_loss,train_psnr) )
            print("===> Epoch {}, Avg_Valid_Loss: {:.4f}, Avg_Valid_PSNR: {:.4f}"
                        .format(epoch, valid_loss,valid_psnr) )

        print('All Done')

    model_path = Path(model_dir+comment+"trained.model")
    torch.save(network.state_dict(),model_path)