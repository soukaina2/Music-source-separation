# Music-source-separation
Our work is based on the U-Net architecture adapted by Jansson et al. in their work on singing voice separationwith convolutional networks: https://ejhumphrey.com/assets/pdf/jansson2017singing.pdf.
* [__Take a look at the demo!__](https://www.colab...)
## Dataset
For our work, we have chosen to take mono files in thefirst place in order to facilitate the architecture and lighten the calculations necessary in the learning phase. In this way, we have chosen to work with the MIR-1K data-set, due to its easeof access and its great length: more than 1000 audio files of about 8 seconds on average, which amounts to 2h15 worth of audio content.We first down-sampled the input audio to8192Hzin orderto speed up processing. To have a fixed input, the data-set was normalized into fragments of 2 seconds-audio. Use the command line: ``` python eval.py``` to convert any wav file into a group 2 seconds audio. 
* [__Here is the data-set prepared__](https://drive.google.com/drive/folders/1sx5rrJ3Y3waKQv_sIh6xtDf0y8x02znl?usp=sharing)
## Usage

* Training
  * ```python train_function.py```
  * Check the loss graph in the chosen path.
* Evaluation
  * ``` python evaluation.py```
  * Check the accuracy in the chosen path.
* To test the model on a sample.
  * ``` python test.py```
  * Before testing, make sure you download [__the checkpoints__](https://drive.google.com/drive/folders/11OaNbx-7b0Z_SMLXtKUiqKIq3_C7QYLv?usp=sharing) of our trained model.
  * Modify the code to choose the sample you want to test.


For more details of the architecture and results, check our [__report__](https://drive.google.com/drive/folders/1VG5ABirrC7aPgKrYopBxy0RjZ8OIqmoL?usp=sharing)
