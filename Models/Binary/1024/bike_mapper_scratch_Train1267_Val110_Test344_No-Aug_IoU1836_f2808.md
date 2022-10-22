Der Trainingsprotzess hiervon war interessant. Die evaluation hat zwar nur 18% IoU ergeben, aber der Trainingsprozess hat folgenden output produziert:

...
Epoch 21: ReduceLROnPlateau reducing learning rate to 9.999999747378752e-06.
1267/1267 [==============================] - 145s 114ms/step - loss: 0.5280 - iou_score: 0.3346 - f1-score: 0.4720 - val_loss: 0.7031 - val_iou_score: 0.1956 - val_f1-score: 0.2969 - lr: 1.0000e-04
Epoch 22/100
1267/1267 [==============================] - ETA: 0s - loss: 0.4937 - iou_score: 0.3671 - f1-score: 0.5063
Epoch 22: val_loss did not improve from 0.70130
Restoring model weights from the end of the best epoch: 17.
1267/1267 [==============================] - 145s 114ms/step - loss: 0.4937 - iou_score: 0.3671 - f1-score: 0.5063 - val_loss: 0.7744 - val_iou_score: 0.1465 - val_f1-score: 0.2256 - lr: 1.0000e-05
Epoch 22: early stopping