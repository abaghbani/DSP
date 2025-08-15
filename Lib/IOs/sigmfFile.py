from fileinput import filename
import numpy as np
import datetime as dt
import sigmf as smf

def readSigmfFile(fileName):

	sig = smf.sigmffile.fromfile(fileName)

	# Get some metadata and all annotations
	sample_rate = sig.get_global_field(smf.SigMFFile.SAMPLE_RATE_KEY)
	sample_count = sig.sample_count
	annotations = sig.get_annotations()
	data = sig.read_samples(0, sample_count)

	print(F'{sample_rate=}, {sample_count=}, data type: {type(data[0])}')
	return (sample_rate, data)

def writeSigmfFile(fileName, data, sample_rate):
	# create data file
	data.tofile(fileName+'.sigmf-data')
	# create the metadata
	meta = smf.SigMFFile(
		global_info = {
			smf.SigMFFile.DATATYPE_KEY: smf.utils.get_data_type_str(data),
			smf.SigMFFile.SAMPLE_RATE_KEY: sample_rate,
			smf.SigMFFile.AUTHOR_KEY: 'Test_SDR.com',
			smf.SigMFFile.DESCRIPTION_KEY: 'this is a test sample file.',
			smf.SigMFFile.VERSION_KEY: smf.__version__,
		}
	)

	# create a capture key at time index 0
	meta.add_capture(0, metadata={
		smf.SigMFFile.FREQUENCY_KEY: 0,
		smf.SigMFFile.DATETIME_KEY: dt.datetime.utcnow().isoformat()+'Z',
	})
	meta.tofile(fileName+'.sigmf-meta') # extension is optional


