## This file should be placed in the root directory of your project.
## Then modify the CMakeLists.txt file in the root directory of your
## project to incorporate the testing dashboard.
## # The following are required to uses Dart and the Cdash dashboard
##   ENABLE_TESTING()
##   INCLUDE(CTest)
set(CTEST_PROJECT_NAME "mtl4")
set(CTEST_NIGHTLY_START_TIME "00:00:00 EST")

set(CTEST_DROP_METHOD "https")
set(CTEST_DROP_SITE "simunova.zih.tu-dresden.de")
set(CTEST_DROP_LOCATION "/CDash/submit.php?project=mtl4")
set(CTEST_DROP_SITE_CDASH TRUE)
set(CTEST_CURL_OPTIONS "CURLOPT_SSL_VERIFYPEER_OFF")
