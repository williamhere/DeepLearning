import argparse
import time
import cv2

from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model', required=True,
                      help='File path of .tflite file.')
  parser.add_argument('-i', '--input', required=False,
                      help='Image to be classified.')
  parser.add_argument('-l', '--labels',
                      help='File path of labels file.')
  parser.add_argument('-k', '--top_k', type=int, default=1,
                      help='Max number of classification results')
  parser.add_argument('-t', '--threshold', type=float, default=0.0,
                      help='Classification score threshold')
  parser.add_argument('-c', '--count', type=int, default=1,
                      help='Number of times to run inference')
  args = parser.parse_args()

  labels = read_label_file(args.labels) if args.labels else {}

  interpreter = make_interpreter(*args.model.split('@'))
  interpreter.allocate_tensors()

  size = common.input_size(interpreter)
  cap = cv2.VideoCapture(0)
  while(True):
    ret , frame = cap.read()
    cv2.imshow('frame',frame)
    image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize(size, Image.ANTIALIAS)
    #image = Image.open(args.input).convert('RGB').resize(size, Image.ANTIALIAS)
    common.set_input(interpreter, image)

    print('----INFERENCE TIME----')
    print('Note: The first inference on Edge TPU is slow because it includes',
          'loading the model into Edge TPU memory.')
    for _ in range(args.count):
      start = time.perf_counter()
      interpreter.invoke()
      inference_time = time.perf_counter() - start
      classes = classify.get_classes(interpreter, args.top_k, args.threshold)
      print('%.1fms' % (inference_time * 1000))

    print('-------RESULTS--------')
    for c in classes:
      print('%s: %.5f' % (labels.get(c.id, c.id), c.score))
    
    if cv2.waitKey(1) &0xFF ==ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
