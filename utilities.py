#just a function to show how much of a loop is done already
def show_loop_progress(counter,length):
	if counter < length - 1 :
		print("%.2f%%"%(100*counter/length) ,end='\r')
	else:
		print("%.2f%%"%(100*counter/length))
	return