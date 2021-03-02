from torch.utils.data import Dataset

class CombinedDataset(Dataset):

    def __init__(self, r_lr, g_lr, b_lr, sar, r_hr, g_hr, b_hr, pytorch = True):
        super().__init__()
        self.files_lr = [self.combine_files_lr(f, g_lr, b_lr, sar) for f in r_lr.iterdir() if not f.is_dir()]
        self.files_hr = [self.combine_files_hr(f, g_hr, b_hr) for f in r_hr.iterdir() if not f.is_dir()]
        self.pytorch = pytorch

    def combine_files_lr(self, r_file: Path, g_lr, b_lr, sar):

        files_lr = {'red_lr':r_file
                ,'green_lr':g_lr/r_file.name.replace('red_lr','green_lr')
                ,'blue_lr':b_lr/r_file.name.replace('red_lr','blue_lr')
                ,'sar':sar/r_file.name.replace('red_lr','sar')}

        return files_lr


    def combine_files_hr(self, r_file: Path, g_hr, b_hr):

        files_hr = {'red_hr':r_file
                ,'green_hr':g_hr/r_file.name.replace('red_hr','green_hr')
                ,'blue_hr':b_hr/r_file.name.replace('red_hr','blue_hr')}

        return files_hr

    def __len__(self):
        
        return len(self.files_lr)

    def open_as_array_lr(self,idx,invert=False):
        raw_lr = np.stack([np.array(Image.open(self.files_lr[idx]['red_lr'])),
                        np.array(Image.open(self.files_lr[idx]['green_lr'])),
                        np.array(Image.open(self.files_lr[idx]['blue_lr'])),
                        np.array(Image.open(self.files_lr[idx]['sar']))
                        ], axis=2)

        if invert:
            raw_lr = raw_lr.transpose((2,0,1))

        return (raw_lr/np.iinfo(raw_lr.dtype).max)

    def open_as_array_hr(self,idx,invert=False):

        raw_hr = np.stack([np.array(Image.open(self.files_hr[idx]['red_hr'])),
                        np.array(Image.open(self.files_hr[idx]['green_hr'])),
                        np.array(Image.open(self.files_hr[idx]['blue_hr']))
                        ], axis=2)
        if invert:
            raw_hr = raw_hr.transpose((2,0,1))

        return (raw_hr/np.iinfo(raw_hr.dtype).max)


    def __getitem__(self,idx):

        x = torch.tensor(self.open_as_array_lr(idx,invert=self.pytorch),dtype=torch.float32)
        y = torch.tensor(self.open_as_array_hr(idx,invert=self.pytorch),dtype=torch.float32)
        #sample = {'lrb': x, 'hrb': y}

        return x,y

    def open_as_pil(self,idx):

        arr = 256*self.open_as_array(idx)

        return Image.fromarray(arr.astype(np.uint8), 'RGB')