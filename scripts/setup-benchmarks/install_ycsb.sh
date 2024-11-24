# Update and install required packages
apt update

# Download YCSB
curl -O --location https://github.com/brianfrankcooper/YCSB/releases/download/0.17.0/ycsb-0.17.0.tar.gz
tar xfvz ycsb-0.17.0.tar.gz
cd ycsb-0.17.0

# Set YCSB_HOME environment variable
export YCSB_HOME=$(pwd)
echo $YCSB_HOME

# Copy contents of ycsb_runner.py to bin/ycsb
cat ycsb_runner.py >bin/ycsb

# Make the ycsb script executable
chmod +x bin/ycsb
