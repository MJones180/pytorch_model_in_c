../output/trained_models/exported_V49_speedup_6_epoch24/model.pt:
	The TorchScript model.
../output/trained_models/exported_V49_speedup_6_epoch24/model.onnx:
	The ONNX model.
../output/trained_models/exported_V49_speedup_6_epoch24/norm_data.txt:
	Contains the normalization info for the model. The lines in order are:
		input max min diff
		input min x
		output max min diff
		output min x
	There is one norm value for all the inputs and a norm value for each of the output values.
../output/trained_models/exported_V49_speedup_6_epoch24/base_field.txt:
	Contains the base field that should be subtracted off. This field of course has only one channel.