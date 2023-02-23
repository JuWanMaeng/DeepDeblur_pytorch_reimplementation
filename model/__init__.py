import os
import re
from importlib import import_module

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

from torch.nn.utils import parameters_to_vector, vector_to_parameters

from . import discriminator


class Model(nn.Module):
    def __init__(self,pretrained=False,load_epoch=0):
        super(Model, self).__init__()

        self.make_d=False
        self.load_epoch=load_epoch
        self.pretrained=pretrained
        self.dtype = torch.float32
        self.save_dir = os.path.join('experiment', 'models')
        os.makedirs(self.save_dir, exist_ok=True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_GPUs = 1
        module = import_module('model.' + 'MSResNet') 

        self.model = nn.ModuleDict()
        self.model.G = module.build_model()      # MSResNet
        
        if self.make_d:
            self.model.D = discriminator.Discriminator()
        else:
            self.model.D=None
        # if self.args.loss.lower().find('adv') >= 0:
        #     self.model.D = Discriminator(self.args)
        # else:
        #     self.model.D = None

        self.to(self.device, dtype=self.dtype, non_blocking=True)
        # self.load(self.load_epoch, path='experiment/models')


    def forward(self, input):
        return self.model.G(input)

    # def parallelize(self):
    #     if self.args.device_type == 'cuda':
    #         if self.args.distributed:
    #             Parallel = DistributedDataParallel
    #             parallel_args = {
    #                 "device_ids": [self.args.rank],
    #                 "output_device": self.args.rank,
    #             }
    #         else:
    #             Parallel = DataParallel
    #             parallel_args = {
    #                 'device_ids': list(range(self.n_GPUs)),
    #                 'output_device': self.args.rank # always 0
    #             }

    #         for model_key in self.model:
    #             if self.model[model_key] is not None:
    #                 self.model[model_key] = Parallel(self.model[model_key], **parallel_args)

    def _save_path(self):
        model_path = os.path.join(self.save_dir, 'best_model.pt')
        return model_path

    def state_dict(self):
        state_dict = {}
        for model_key in self.model:
            if self.model[model_key] is not None:
                parallelized = isinstance(self.model[model_key], (DataParallel, DistributedDataParallel))
                if parallelized:
                    state_dict[model_key] = self.model[model_key].module.state_dict()
                else:
                    state_dict[model_key] = self.model[model_key].state_dict()

        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        for model_key in self.model:
            parallelized = isinstance(self.model[model_key], (DataParallel, DistributedDataParallel))
            if model_key in state_dict:
                if parallelized:
                    self.model[model_key].module.load_state_dict(state_dict[model_key], strict)
                else:
                    self.model[model_key].load_state_dict(state_dict[model_key], strict)

    def save(self):
        torch.save(self.state_dict(), self._save_path())

    def load(self, epoch=None, path=None):
        if path:
            model_name = path
        elif isinstance(epoch, int):
            if epoch < 0:
                epoch = self.get_last_epoch()
            if epoch == 0:   # epoch 0
                # make sure model parameters are synchronized at initial
                # for multi-node training (not in current implementation)
                # self.synchronize()

                return  # leave model as initialized

            model_name = self._save_path(epoch)
        else:
            raise Exception('no epoch number or model path specified!')

        print('Loading model from {}'.format(model_name))
        state_dict = torch.load(model_name, map_location=self.device)
        self.load_state_dict(state_dict)

        return

    # def synchronize(self):
    #     if self.args.distributed:
    #         # synchronize model parameters across nodes
    #         vector = parameters_to_vector(self.parameters())

    #         dist.broadcast(vector, 0)   # broadcast parameters to other processes
    #         if self.args.rank != 0:
    #             vector_to_parameters(vector, self.parameters())

    #         del vector

    #     return

    def get_last_epoch(self):
        model_list = sorted(os.listdir(self.save_dir))
        if len(model_list) == 0:
            epoch = 0
        else:
            epoch = int(re.findall('\\d+', model_list[-1])[0]) # model example name model-100.pt

        return epoch

    def print(self):
        print(self.model)

        return
