from . import pet_dataset


def get_datasets(opts):
    if opts.dataset == 'PET':
        trainset = pet_dataset.PET_Train(opts)
        valset = pet_dataset.PET_Test(opts)

    elif opts.dataset == 'XXX':
        a = 1
        # trainset = sv_dataset.SVTrain(opts.data_root)
        # valset = sv_dataset.SVTest(opts.data_root)

    return trainset, valset
