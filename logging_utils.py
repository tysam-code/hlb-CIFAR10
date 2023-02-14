from functools import partial
import platform

import torch

# We basically need to look up local variables by name so we can have the names, so we can pad to the proper column width.
# Printing stuff in the terminal can get tricky and this used to use an outside library, but some of the required stuff seemed even
# more heinous than this, unfortunately. So we switched to the "more simple" version of this!

def format_for_table(x, vars): return (f"{vars[x]}".rjust(len(x))) \
    if type(vars[x]) == int else "{:0.4f}".format(vars[x]).rjust(len(x)) \
    if vars[x] is not None \
    else " "*len(x)

logging_columns_list = ['epoch', 'train_loss', 'val_loss',
                        'train_acc', 'val_acc', 'ema_val_acc', 'train_time']

# Print out our training details (sorry for the complexity, the whole logging business here is a bit of a hot mess once the columns need to be aligned and such....)
# define the printing function and print the column heads
def print_training_details(columns_list=logging_columns_list, separator_left='|  ',
                           separator_right='  ', final="|", column_heads_only=False,
                           is_final_entry=False, vars=None):

    if vars is not None:
        columns_list = list(map(
                                partial(format_for_table, vars=vars), logging_columns_list)
                            )

    print_string = ""
    if column_heads_only:
        for column_head_name in columns_list:
            print_string += separator_left + column_head_name + separator_right
        print_string += final
        print('-'*(len(print_string)))  # print the top bar
        print(print_string)
        print('-'*(len(print_string)))  # print the bottom bar
    else:
        for column_value in columns_list:
            print_string += separator_left + column_value + separator_right
        print_string += final
        print(print_string)
    if is_final_entry:
        print('-'*(len(print_string)))  # print the final output bar

def print_headers():
    # print out the training column heads before we print the actual content for each run.
    print_training_details(column_heads_only=True)

def print_device_info(device_name):
    device = torch.device(device_name)
    print('Using device:', device)
    print()
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('GPU Memory Allocated/Cached:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB',
              "//", round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    elif device.type == 'cpu':
        print("CPU: ", platform.processor())
