# exploring-mnist-mix

Several explorations and experiments centered around the MNIST-MIX ([arXiv](https://arxiv.org/abs/2004.03848), [GitHub](https://github.com/jwwthu/MNIST-MIX)) dataset, focusing mainly on comparing feedforward and convolutional networks and examining the effects of various data manipulations on learning.

Final project for CS 445 / 545 at Portland State University, Spring 2020.

Created and maintained by:
- [Umair Khan](https://github.com/khanu263)
- [Damon Aliomrany](https://github.com/domrany64)
- [Mi Yon Kim](https://github.com/youn0125)
- [Grant Baker](https://github.com/gnbpdx)
- [Bach Khuat](https://github.com/bachkhuat)

### Using `main.py`

Example -- create a new feedforward network, train and save it.

```
python main.py --create feedforward 100 100 100
               --labels specific
               --train data/splits/all_train.split 10 16 0.01 0.9
               --test data/splits/all_test.split tests/10_3x100.txt
               --log logs/10_3x100.log
               --save models/10_3x100.pt
               --gpu
```

Example -- load a ResNet model and test it on English.

```
python main.py --load models/10_resnet14.pt
               --labels specific
               --test data/splits/3_english_test.split tests/10_resnet14_english.txt
               --gpu
```

- `--create [type] [parameters]` -- Use to build and train a new model. The type is either `feedforward` or `resnet` and the parameters are a sequence of numbers defining the model. For example, to create a feedforward model with three layers of 100 neurons each, use `--create feedforward 100 100 100`. Alternatively, to create an 8-layer ResNet, user `--create resnet 1 1 1`. (ResNet layers are calculated as `2 * sum(blocks) + 2`. In this implementation, there are three block sizes to specify.)
- `--load [path]` -- Use to load a PyTorch model already saved to disk. This cannot be used simultaneously with the `--create` flag.
- `--labels [specific/agnostic]` -- Use this to specify whether image labels should be "specific" (100 classes) or "agnostic" (10 classes) to the language.
- `--train [path] [epochs] [batch size] [learning rate] [momentum]` -- Use this to train a model. The path is to the split file, which you should create in the data folder.
- `--test [data path] [save path]` -- Use this to test a model at the end of the script. The first path is to the split file to use. The second path is optional, and if used will specify the file to save the confusion matrix and accuracy to. If `--train` is used, then this must be used; however you can use `--test` on its own.
- `--log [path]` -- Use this to save a file containing the training loss, validation loss, and accuracy at each epoch of training.
- `--save [path]` -- Use this to save the PyTorch model to the specified path at the end of the script.
- `--gpu` -- A binary flag, include it to train with CUDA.

### Data summary

| **ID** | **Language** | **Train** | **Test** | **Total** |
|--------|--------------|-----------|----------|-----------|
| 0      | Arabic       | 62400     | 10600    | 73000     |
| 1      | Bangla       | 39990     | 9150     | 49140     |
| 2      | Devanagari   | 2400      | 600      | 3000      |
| 3      | English      | 240000    | 40000    | 280000    |
| 4      | Farsi        | 60000     | 20000    | 80000     |
| 5      | Kannada      | 60000     | 20240    | 80240     |
| 6      | Swedish      | 6600      | 1000     | 7600      |
| 7      | Telegu       | 2400      | 600      | 3000      |
| 8      | Tibetan      | 14214     | 3554     | 17768     |
| 9      | Urdu         | 6606      | 1414     | 8020      |