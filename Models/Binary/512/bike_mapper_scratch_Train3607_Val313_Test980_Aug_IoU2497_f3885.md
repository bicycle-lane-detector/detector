used augmentation 

```python
augmentation = {

"rotation_range": 90,
"width_shift_range": 0.2,
"height_shift_range": 0.2,
"fill_mode": "nearest",
# "cval": 0,
"horizontal_flip": "True",
"vertical_flip": "True",
"validation_split": 0.08
}
```

worse IOU but validation and training do not diverge as strongly