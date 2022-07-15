# genformer_public

see *gs://genformer_data/data* for basenji barnyard data reprocessed to sequence length 196k, 128 bp resolution. see targets.txt file in human and mouse folders respectiely for metadata for the datasets in the TFRs.

Note: each sequence has an added 10 bp to allow the sequence shifts for train time data augmentation in the original Enformer paper. 

data from:

"Effective gene expression prediction from sequence by integrating long-range interactions"

Žiga Avsec, Vikram Agarwal, Daniel Visentin, Joseph R. Ledsam, Agnieszka Grabska-Barwinska, Kyle R. Taylor, Yannis Assael, John Jumper, Pushmeet Kohli, David R. Kelley
