from rec_net.network import *
from rec_net.new_network import *

def make_model(rank, args):
    module = __import__('rec_net')
    class_ = getattr(module, args.model)
    print(class_.__name__)
    model = class_(rank, args)
    return model
