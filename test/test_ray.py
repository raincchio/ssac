import ray
import logging
ray.init(
    # If true, then output from all of the worker processes on all nodes will be directed to the driver.
    log_to_driver=True,
    logging_level=logging.INFO,

    # The amount of memory (in bytes)
    object_store_memory=1073741824, # 1g
    redis_max_memory=1073741824 # 1g
)
print(f'{bcolors.FAIL}pass{bcolors.ENDC}')