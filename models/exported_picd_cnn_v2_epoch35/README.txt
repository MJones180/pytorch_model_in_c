../output/trained_models/exported_picd_cnn_v2_epoch35/model.pt:
	The TorchScript model.
../output/trained_models/exported_picd_cnn_v2_epoch35/model.onnx:
	The ONNX model.
../output/trained_models/exported_picd_cnn_v2_epoch35/norm_data.txt:
	Contains the normalization info for the model. The lines in order are:
		input max min diff
		input min x
		output max min diff
		output min x
	There is one norm value for all the inputs and a norm value for each of the output values.
../output/trained_models/exported_picd_cnn_v2_epoch35/base_field.txt:
	Contains the base field that should be subtracted off. This field of course has only one channel.
../output/trained_models/exported_picd_cnn_v2_epoch35/example_data/first_input_row_norm.txt:
	Example input row after norm is done and base field is subtracted.
../output/trained_models/exported_picd_cnn_v2_epoch35/example_data/first_output_row_norm_truth.txt:
	Example truth output row before denorm is done.
../output/trained_models/exported_picd_cnn_v2_epoch35/example_data/first_output_row_truth.txt:
	Example truth output row after denorm is done.
../output/trained_models/exported_picd_cnn_v2_epoch35/example_data/first_output_row_norm_ts.txt:
	Example TorchScript output row before denorm is done.
../output/trained_models/exported_picd_cnn_v2_epoch35/example_data/first_output_row_ts.txt:
	Example TorchScript output row after denorm is done.
../output/trained_models/exported_picd_cnn_v2_epoch35/example_data/first_output_row_norm_onnx.txt:
	Example ONNX output row before denorm is done.
../output/trained_models/exported_picd_cnn_v2_epoch35/example_data/first_output_row_onnx.txt:
	Example ONNX output row after denorm is done.