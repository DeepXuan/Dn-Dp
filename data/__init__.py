'''create dataset and dataloader'''
import logging
from re import split
import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase == 'val':
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, dataset_name, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    if dataset_name=='uint8dataset':
        from data.LRHR_dataset import LRHRDataset as D
    elif dataset_name=='uint16dataset':
        from data.LRHR_dataset_new import LRHRDatasetCT as D
    elif dataset_name=='uint16dataset_dn':
        from data.LRHR_dataset_new import NoisyCleanDatasetCT as D
    else:
        logging.raiseExceptions('unvalid dataset type')
    if dataset_name == 'uint16dataset_patch':
        dataset = D(dataroot=dataset_opt['dataroot'],
            datatype=dataset_opt['datatype'],
            l_resolution=dataset_opt['l_resolution'],
            r_resolution=dataset_opt['r_resolution'],
            patch_size=dataset_opt['patch_size'],
            patch_sample=dataset_opt['patch_sample'],
            split=phase,
            data_len=dataset_opt['data_len'],
            need_LR=(mode == 'LRHR')
            )
    else:
        dataset = D(dataroot=dataset_opt['dataroot'],
                    # dataroot1=dataset_opt['dataroot1'],
                    datatype=dataset_opt['datatype'],
                    exclude_patients=dataset_opt['exclude_patients'], 
                    include_patients=dataset_opt['include_patients'],
                    l_resolution=dataset_opt['l_resolution'],
                    r_resolution=dataset_opt['r_resolution'],
                    patch_n=dataset_opt['patch_n'],
                    split=phase,
                    data_len=dataset_opt['data_len'],
                    need_LR=(mode == 'LRHR')
                    )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
