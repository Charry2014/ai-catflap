# This is a minimal hacky script to take the unzipped output from Label Studio and make
# a Tensor Flow Lite model out of them. It uses the object_detector method to create a 
# model based on efficientdet_lite0. 
# The versions specified in the requirements.txt are really the newest versions that work.
import os
import glob
import sys
import json
import re
import datetime
import shutil

print("Starting at ", datetime.datetime.now())

from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

from PIL import Image

def newestproject(path:str):
  # project-1-at-2023-03-14-10-08-23f5de9f
  '''Finds the newest project folder'''
  retval = path
  if re.search('project-', path) == None:
    list = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and re.search('project-', d) != None]
    if len(list) == 0:
       print("Error: No project data found in " + path)
       sys.exit(1)
    retval = max(list, key=os.path.getmtime)
    print(f'Processing directory {retval}')
  
  return retval

def fix_xml(filename:str):
  with open(filename, 'r') as f:
      data = f.read().splitlines(True)
      with open(filename, 'w') as fout:
        for l in data:
          if l.startswith('<?xml '):
            continue
          l = l.replace(".png", ".jpg")
          fout.write(l)

def convert_png_2_jpg(filename:str, outdir:str):
  with Image.open(filename) as f:
      rgb_im = f.convert('RGB')
      outfile = os.path.join(outdir, os.path.splitext(os.path.basename(filename))[0]+'.jpg')
      rgb_im.save(outfile)


def preprocess_files(args):
  # Copy the XML annotations into the base dir and fix them up
  src = os.path.join(args.base_path, args.annotations_dir)
  if os.path.exists(src):
    for filename in glob.glob(os.path.join(src,'*.xml')):
      shutil.copy(filename, args.base_path)
      fix_xml(os.path.join(args.base_path, os.path.basename(filename)))

  # Similarly copy the images, and convert png to jpeg
  src = os.path.join(args.base_path, args.images_dir)
  if os.path.exists(src):
    for filename in glob.glob(os.path.join(src,'*.png')):
      convert_png_2_jpg(filename, args.base_path)
    for filename in glob.glob(os.path.join(src,'*.jpg')):
      shutil.copy(filename, args.base_path)
   

def main():
    import argparse
    parser = argparse.ArgumentParser(description="export2tflite - ´Exported images in Pascal XML VOC convert to TensorFlow Lite format")
    parser.add_argument(
        '--base_path',
        help="Stream source link",
        default='~/Downloads/',
        type=str, action='store', required=False)
    parser.add_argument(
        '--output_file',
        help="Filename for the output model",
        default='../../test/cats.tflite',
        type=str, action='store', required=False)
    parser.add_argument(
        '--labels',
        help="Filename for the labels in JSON",
        default='../../src/labels.json',
        type=str, action='store', required=False)
    parser.add_argument(
        '--annotations_dir',
        help="Location of XML annotations",
        default='Annotations',
        type=str, action='store', required=False)
    parser.add_argument(
        '--images_dir',
        help="Location of images",
        default='images',
        type=str, action='store', required=False)
    args = parser.parse_args()

    # Clean up the incoming files from Label Studio
    args.base_path = newestproject(os.path.expanduser(args.base_path))
    preprocess_files(args)

    with open(os.path.expanduser(args.labels), 'r') as f:
        args.labels = json.load(f)['labels']
    print('Labels: ' + str(args.labels))

    train_data = object_detector.DataLoader.from_pascal_voc(
        args.base_path,
        args.base_path,
        args.labels
    )

    val_data = object_detector.DataLoader.from_pascal_voc(
        '.',
        '.',
        args.labels
    )

    spec = model_spec.get('efficientdet_lite0')
    # This seems to need an internet connection, returns SSL error if something goes wrong. Turns out that Python needs some sort of certs
    # Exception has occurred: URLError
    # <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1129)>
    # ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1129)
    # https://stackoverflow.com/questions/50236117/scraping-ssl-certificate-verify-failed-error-for-http-en-wikipedia-org
    # /Applications/Python 3.9/Install Certificates.command
    model = object_detector.create(train_data, model_spec=spec, batch_size=4, train_whole_model=True, epochs=20, validation_data=val_data)
    model.export(export_dir='.', tflite_filename='cats.tflite')
    print("Done at ", datetime.datetime.now())

if __name__ == "__main__":
  main()