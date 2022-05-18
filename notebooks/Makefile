default: BASICS

J=jupyter nbconvert  --ExecutePreprocessor.timeout=0 --allow-errors --execute
# J=jupyter nbconvert  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=0 --allow-errors --execute
JN=$(J) --to notebook --inplace

BASICS: B_01 B_02 B_03 B_04 

B_01:
	$(JN) BASICS_01_timesurface.ipynb
	git commit -m 'results basics 01'
	git push
B_02:
	$(JN) BASICS_02_hots.ipynb
	git commit -m 'results basics 02'
	git push
B_03:
	$(JN) BASICS_03_hotsline.ipynb
	git commit -m 'results basics 03'
	git push
B_04:
	$(JN) BASICS_04_robustness.ipynb
	git commit -m 'results basics 04'
	git push
    
RESULTS_NMNIST:
	$(JN) RESULTS_NMNIST.ipynb
	git commit -m 'results nmnist'
	git push