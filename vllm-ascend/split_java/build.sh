cd /path/to/SplitLwdCheckpoint.java        # 该目录下只有这个 .java 就行
mkdir -p build
javac -d build SplitLwdCheckpoint.java     # .class 全部落在 build/,不污染当前目录
jar cfe split-lwd-checkpoint.jar SplitLwdCheckpoint -C build .   # Java 8 也能用的写法
rm -rf build                               # 收尾:只留 .java 和 .jar
