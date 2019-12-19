#!/usr/bin/env python3
from hyperion import Hyperion
import sys

def main(*args):
	ipbase = "10.162.34.{}"

	for i in range(2,256):
		try:
			h = Hyperion(ipbase.format(i))
			h.serial_number
			break

		# try

		except OSError as e:
#			print (e)
			continue

	# for

	print("IP found @ ", ipbase.format(i))
	return(ipbase.format(i))

# main

if __name__ == "__main__":
	main(sys.argv[1:])
