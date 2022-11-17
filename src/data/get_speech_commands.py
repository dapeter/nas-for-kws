import os
import tarfile
from pathlib import Path
import re
import hashlib
import wave
import struct
import urllib.request
from tqdm import tqdm


dataset_url = 'http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz'
dataset_name = 'speech_commands_v0.01'
keywords = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
unknowns = ['bed', 'cat', 'eight', 'four', 'house', 'nine', 'seven', 'six', 'tree', 'wow',
            'bird', 'dog', 'five', 'happy', 'marvin', 'one', 'sheila', 'three', 'two', 'zero']
background_folder_name = '_background_noise_'
output_path = '../../data/'


def which_set(filename, validation_percentage=10, testing_percentage=10):
    MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134M

    """Determines which data partition the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.

    Args:
      filename: File path of the data sample.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      String, one of 'training', 'validation', or 'testing'.
    """
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result


def which_files(members, partition):
    for tarinfo in members:
        if os.path.splitext(tarinfo.name)[1] == '.wav' and which_set(tarinfo.name) == partition:
            class_name = os.path.basename(os.path.dirname(tarinfo.name))
            if class_name in keywords or class_name == background_folder_name:
                yield tarinfo
            elif class_name in unknowns:
                tarinfo.path = './unknown/' + os.path.splitext(os.path.basename(tarinfo.path))[0] \
                               + "_" + class_name + ".wav"
                yield tarinfo
            else:
                assert RuntimeError('{} caused error.'.format(tarinfo.name))


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_from_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


if __name__ == '__main__':
    os.chdir(output_path)

    dataset_tar_name = os.path.basename(dataset_url)
    if not os.path.exists(dataset_tar_name):
        print('Downloading {} ...'.format(dataset_tar_name))
        download_from_url(dataset_url, dataset_tar_name)

    raw_dataset_save_path = '{}/'.format(dataset_name)
    if not os.path.exists(raw_dataset_save_path):
        for dataset_type in ['training', 'testing', 'validation']:
            with tarfile.open(dataset_tar_name, 'r:gz') as tar:
                save_path = raw_dataset_save_path + dataset_type + '/'
                print('Extracting {} files to {} ...'.format(dataset_type, save_path))
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, path=save_path, members=which_files(tar,dataset_type))

                silence_save_path = save_path + 'silence/'
                sr = 16000
                Path(silence_save_path).mkdir()
                print('Creating silence.wav in {} ...'.format(silence_save_path))
                with wave.open(silence_save_path + 'silence.wav', 'w') as wavfile:
                    wavfile.setnchannels(1)
                    wavfile.setframerate(sr)
                    wavfile.setsampwidth(2)
                    for i in range(sr):
                        data = struct.pack('<h', 0)
                        wavfile.writeframesraw(data)

    print("Done.")
