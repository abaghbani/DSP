import os
import platform

def get_file_from_path(path, extension='.bin', def_file=0):
	logFolderContent = os.listdir(path)
	logFolderContent = [f for f in logFolderContent if f.endswith(extension)]
	# logFolderContent.reverse()
	for i in range(len(logFolderContent)):
		if def_file == i:
			print(f'{i}> {logFolderContent[i]} (*)')
		else:
			print(f'{i}> {logFolderContent[i]}')
	file_num = input('Select a file [0, 1, ...] = ')
	file_num = def_file if file_num == '' else int(file_num)
	print(logFolderContent[file_num])
	return path+logFolderContent[file_num]

def get_console_key():
	print('Press a command:')
	if "Windows" in platform.system():
		import msvcrt
		while True:
			if msvcrt.kbhit():
				break
		key = msvcrt.getch().decode("utf-8")
		print(key)
		key = key.lower()
		return key
	else:
		while True:
			key = input()
			if key.isascii:
				break
		key = key.lower()
		return key[0]
