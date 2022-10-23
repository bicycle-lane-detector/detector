Followed results from this paper https://arxiv.org/pdf/2002.08438v1.pdf
freezing right half of the convolutional layer (layers ["conv2d_9","conv2d_10","conv2d_11","conv2d_12","conv2d_13","conv2d_14","conv2d_15","conv2d_16","conv2d_17"] )

resulting in 
Total params: 1,946,993
Trainable params: 765,969
Non-trainable params: 1,181,024

lr 1e-4

started training and got 25% IoU right off the bat

training iou_score was at 41 %
val_iou_score only at 31%
Overfitting?
reordering validationset didn't help