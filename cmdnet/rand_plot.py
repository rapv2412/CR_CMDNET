class rand_plot():
    @staticmethod
    def plot_all(names, r=random.randint(0, 9),figsize = (9,9)):
        
        fig,axs = plt.subplots(len(names),4,figsize=figsize)
        p = 0 
        for name in names:
            for n in range(3):
                color = name[r][n].detach().cpu().numpy()
                axs[p][n].imshow(color)
            rgb = np.dstack(
                [name[r][0].detach().cpu().numpy(),
                name[r][1].detach().cpu().numpy(),
                name[r][2].detach().cpu().numpy()])
            axs[p][3].imshow(rgb)
            p += 1
        plt.show()

    def plot_rgb(names,r=random.randint(0, 9), figsize = (9,9)):
        fig,axs = plt.subplots(1,len(names)+1,figsize=figsize)
        p = 1
        sar = names[0][r][3].detach().cpu().numpy()
        axs[0].imshow(sar,cmap='gray')
        for name in names:
            rgb = np.dstack(
                [name[r][0].detach().cpu().numpy(),
                name[r][1].detach().cpu().numpy(),
                name[r][2].detach().cpu().numpy()])
            axs[p].imshow(rgb)
            p += 1
        plt.show()