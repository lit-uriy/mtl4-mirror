mkdir -p mtl_dep_repo/main
cp standard/*.deb mtl_dep_repo/main/
cp examples/*.deb mtl_dep_repo/main/
cp tests/*.deb mtl_dep_repo/main/
cp all/*.deb mtl_dep_repo/main/

cd mtl_dep_repo
apt-ftparchive packages main/ | gzip -9c > main/Packages.gz
