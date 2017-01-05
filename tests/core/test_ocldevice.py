"""


mweigert@mpi-cbg.de

"""


from __future__ import print_function, unicode_literals, absolute_import, division
from gputools import get_device, init_device

def test_devices():
    print("current device:")
    get_device().print_info()

    # print("best device:")
    # init_device(id_platform = -1,
    #             id_device = -1,
    #             )
    # get_device().print_info()


if __name__ == '__main__':


    pass