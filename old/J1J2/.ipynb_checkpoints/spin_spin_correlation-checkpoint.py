{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "00af0f42",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Tensorflow version = 1.13.1\n",
      "\n",
      "WARNING: The TensorFlow contrib module will not be included in TensorFlow 2.0.\n",
      "For more information, please see:\n",
      "  * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md\n",
      "  * https://github.com/tensorflow/addons\n",
      "If you depend on functionality not listed there, please file an issue.\n",
      "\n",
      "WARNING:tensorflow:From C:\\Users\\Administrator\\anaconda3\\envs\\RNN\\lib\\site-packages\\tensorflow\\contrib\\cudnn_rnn\\python\\ops\\cudnn_rnn_ops.py:116: GRUCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "This class is equivalent as tf.keras.layers.GRUCell, and will be replaced by that in Tensorflow 2.0.\n",
      "WARNING:tensorflow:From C:\\Users\\Administrator\\PycharmProjects\\RNNWavefunctions\\J1J2\\ComplexRNNwavefunction.py:40: MultiRNNCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "This class is equivalent as tf.keras.layers.StackedRNNCells, and will be replaced by that in Tensorflow 2.0.\n",
      "[name: \"/device:CPU:0\"\n",
      "device_type: \"CPU\"\n",
      "memory_limit: 268435456\n",
      "locality {\n",
      "}\n",
      "incarnation: 9719104400240995096\n",
      "]\n",
      "WARNING:tensorflow:From C:\\Users\\Administrator\\anaconda3\\envs\\RNN\\lib\\site-packages\\tensorflow\\python\\framework\\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Colocations handled automatically by placer.\n",
      "one_hot_samples: Tensor(\"one_hot_1:0\", shape=(?, 96, 2), dtype=float32, device=/device:CPU:0)\n",
      "cond_prob: Tensor(\"Sum_2:0\", shape=(?, 96), dtype=float32, device=/device:CPU:0)\n",
      "(130, 256)\n",
      "33280\n",
      "(256,)\n",
      "256\n",
      "(2, 128)\n",
      "256\n",
      "(128, 128)\n",
      "16384\n",
      "(128,)\n",
      "128\n",
      "(128,)\n",
      "128\n",
      "(128, 2)\n",
      "256\n",
      "(2,)\n",
      "2\n",
      "(128, 2)\n",
      "256\n",
      "(2,)\n",
      "2\n",
      "50948\n",
      "Loading the model from checkpoint\n",
      "WARNING:tensorflow:From C:\\Users\\Administrator\\anaconda3\\envs\\RNN\\lib\\site-packages\\tensorflow\\python\\training\\saver.py:1266: checkpoint_exists (from tensorflow.python.training.checkpoint_management) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Use standard file APIs to check for files with this prefix.\n",
      "INFO:tensorflow:Restoring parameters from ../Check_Points/J1J2/RNNwavefunction_N96_samp128_lradap0.0005_complexGRURNN_J1J20.25_units_128_zeromag.ckpt\n",
      "Loading was successful!\n"
     ]
    }
   ],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
