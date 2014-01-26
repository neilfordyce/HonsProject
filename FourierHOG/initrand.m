function initrand()
try
  RandStream.setGlobalStream(RandStream('mt19937ar','seed',3));
catch
  rand('twister',3);
end
end