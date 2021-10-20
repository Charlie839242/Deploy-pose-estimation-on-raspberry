import tensorflow as tf

classifier = tf.lite.Interpreter(model_path='./model/classifier.tflite', num_threads=4)
print('model loaded successfully')

classifier.allocate_tensors()

# 获取模型的详细数据
input_details = classifier.get_input_details()
output_details = classifier.get_output_details()

print(input_details)
print(output_details)