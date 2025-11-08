CC = gcc
CFLAGS = -O2 -Wall
TARGET = gol-cpu

all: $(TARGET)

$(TARGET): gol-cpu.c
	$(CC) $(CFLAGS) -o $(TARGET) gol-cpu.c

# Numbers 20, 40 and 10 indicate default parameters, meaning 20 rows, 40 columns and 10 steps
run: $(TARGET)
	./$(TARGET) 20 40 10

clean:
	rm -f $(TARGET)
