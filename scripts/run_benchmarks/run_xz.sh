sudo apt-get update
sudo apt-get install -y xz-utils

dd if=/dev/urandom of=testfile bs=1M count=1024 # Creates 1GB file

# Run compression benchmark
echo "Running compression test..."
time xz -9 -T $(nproc) testfile

# Run decompression benchmark
echo "Running decompression test..."
time xz -d -T $(nproc) testfile.xz
