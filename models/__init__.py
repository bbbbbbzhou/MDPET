from models import model_svrhd_dp_gan


def create_model(opts):
    if opts.model_type == 'model_svrhd_dp_gan':
        model = model_svrhd_dp_gan.RegModel(opts)

    else:
        raise NotImplementedError

    return model
