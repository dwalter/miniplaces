for i in range(10001):
	num_str = str(i)
	while len(num_str) < 8:
		num_str = '0' + num_str
	print 'test/' + num_str + '.jpg'