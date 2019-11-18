input_size = 4
kernel_size = 4
stride = 4
pad = 0

deconv_size = (input_size-1)*stride -2*pad +kernel_size
print(deconv_size)

conv_size = ((input_size +2*pad - kernel_size)/stride) +1
print(conv_size)