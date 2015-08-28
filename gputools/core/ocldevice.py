'''
@author: mweigert

A basic wrapper class around pyopencl to handle image manipulation via OpenCL
basic usage:


    #create a device
    dev = OCLDevice(useGPU=True, useDevice = 0, printInfo = True)

'''

import logging
logger = logging.getLogger(__name__)

import pyopencl

class OCLDevice:
    """ a wrapper class representing a CPU/GPU device"""

    def __init__(self,initCL = True, **kwargs):
        """ same kwargs as initCL """
        if initCL:
            self.init_cl(**kwargs)

    def init_cl(self,useDevice = 0, useGPU = True, printInfo = False, context_properties= None):
        platforms = pyopencl.get_platforms()
        if len(platforms) == 0:
            raise Exception("Failed to find any OpenCL platforms.")
            return None

        devices = []
        if useGPU:
            devices = platforms[0].get_devices(pyopencl.device_type.GPU)
            if len(devices) == 0:
                logger.warning("Could not find GPU device...")
        else:
            devices = platforms[0].get_devices(pyopencl.device_type.CPU)
            if len(devices) == 0:
                logger.warning("Could neither find GPU nor CPU device....")

        if len(devices) ==0:
            logger.warning("couldnt find any devices...")
            return None
        else:
            logger.info("using device: %s"%devices[useDevice].name)

        # Create a context using the nth device
        self.context = pyopencl.Context(devices = [devices[useDevice]],properties = context_properties)

        self.device =  devices[useDevice]

        self.queue = pyopencl.CommandQueue(self.context,properties = pyopencl.command_queue_properties.PROFILING_ENABLE)

        if printInfo:
            self.printInfo()


    def print_info(self):
        platforms = pyopencl.get_platforms()
        print("\n-------- available devices -----------")
        for p in platforms:
            print(("platform: \t",p.name))
            printNames = [["CPU",pyopencl.device_type.CPU],
                          ["GPU",pyopencl.device_type.GPU]]
            for name, identifier in printNames:
                print(("device type: \t" , name))
                try:
                    for d in p.get_devices(identifier):
                        print(("\t ", d.name))
                except:
                    print(("nothing found: ", name))

        infoKeys = ['NAME','GLOBAL_MEM_SIZE',
                    'GLOBAL_MEM_SIZE','MAX_MEM_ALLOC_SIZE',
                    'LOCAL_MEM_SIZE','IMAGE2D_MAX_WIDTH',
                    'IMAGE2D_MAX_HEIGHT','IMAGE3D_MAX_WIDTH',
                    'IMAGE3D_MAX_HEIGHT','IMAGE3D_MAX_DEPTH',
                    'MAX_WORK_GROUP_SIZE','MAX_WORK_ITEM_SIZES']

        print("\n-------- currently used device -------")

        for k in infoKeys:
            print(("%s: \t  %s"% (k, self.get_info(k))))



    def get_info(self, info_str = "MAX_MEM_ALLOC_SIZE"):
        return self.device.get_info(getattr(pyopencl.device_info,info_str))



if __name__ == '__main__':
    OCLDevice().print_info()
