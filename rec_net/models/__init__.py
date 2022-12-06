from rec_net.models import du_recurrent_model
from rec_net.models import unet


def create_model(opts):
    if opts.model_type == 'model_recurrent_dual':
        model = du_recurrent_model.RecurrentModel(opts)

    else:
        raise NotImplementedError

    return model
