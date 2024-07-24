import torch
import numpy as np
import xarray as xr


class ForwardModelDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path

        self.ids_list = range(0, 1342)

        self.perm_mean = -1.3754382634162903 #3.5748687267303465
        self.perm_std = 3.6160271644592283 #4.6395333366394045
        # self.por_mean = 0.09433708190917969
        # self.por_std = 0.03279830865561962


        self.pressure_mean = 59.839262017194635 # 336.26219848632815
        self.pressure_std = 16.946480998339133 #130.7361669921875
        self.pressure_min = 51.5
        self.pressure_max = 299.43377685546875
        self.co2_mean = 0.9997552563110158 #0.03664950348436832
        self.co2_std = 0.004887599662507033 #0.13080736815929414
        self.U_z_min = -0.03506183251738548
        self.U_z_max = -7.1078920882428065e-06
        

        self.ft_min = 6.760696180663217e-08 #0.0
        self.ft_max = 0.0009333082125522196 #28074.49609375



    def __len__(self):
        return len(self.ids_list)

    def __getitem__(self, idx):

        data = xr.load_dataset(f'{self.path}_{self.ids_list[idx]}.nc')

        state = np.concat((data['PRESSURE'].data, data['H2O'].data, data['U_z'].data), axis=1)
        
        pars = data['Perm'].data[0] # np.stack((data['Perm'].data, data['Por'].data), axis=0)
        ft = data['c_0_rate'].data[0]


        state = torch.tensor(state, dtype=torch.float32)
        pars = torch.tensor(pars, dtype=torch.float32)
        pars = torch.log(pars)
        ft = torch.tensor(ft, dtype=torch.float32)

        state = torch.permute(state, (1, 2, 3, 0))

        # state[0] = (state[0] - self.pressure_mean) / self.pressure_std

        state[0] = (state[0] - self.pressure_min) / (self.pressure_max - self.pressure_min)
        state[2] = (state[2] - self.U_z_min) / (self.U_z_max - self.U_z_min)
        #state[1] = (state[1] - self.co2_mean) / self.co2_std
        pars[0] = (pars[0] - self.perm_mean) / self.perm_std
        #pars[1] = (pars[1] - self.por_mean) / self.por_std

        ft = (ft - self.ft_min) / (self.ft_max - self.ft_min)


        return state, pars, ft
    

class ParsDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path

        self.ids_list = range(0, 1342)

        self.min = 1e8
        self.max = 0
        
        self.perm_mean = 2.5359260627303146#-1.3754382634162903 #3.5748687267303465
        self.perm_std = 2.1896950964067803#3.6160271644592283 #4.6395333366394045

        #self.por_mean = 0.09433708190917969
        #self.por_std = 0.03279830865561962


    def __len__(self):
        return len(self.ids_list)

    def __getitem__(self, idx):

        data = xr.load_dataset(f'{self.path}_{self.ids_list[idx]}.nc')

        #pars = np.stack((data['Perm'].data, data['Por'].data), axis=0)
        pars = data['Perm'].data[0]
        # pars = data['U_z'].data[0]

        pars = torch.tensor(pars, dtype=torch.float32)
        pars = torch.log(pars)

        pars[0] = (pars[0] - self.perm_mean) / self.perm_std

        return pars