CC=gcc
CFLAGS=-Wall -Wextra -pedantic -std=c99 -Werror -g -pg -O0 -fsanitize=address -fsanitize=undefined -Wshadow

equations : 3-equations.c functions.o gauss_jordan.o log.o
	$(CC) $(CFLAGS) -o $@ $< functions.o gauss_jordan.o log.o

functions.o : functions.c functions.h log.o
	$(CC) $(CFLAGS) -c $< log.o

gauss_jordan.o : gauss_jordan.c gauss_jordan.h log.o
	$(CC) $(CFLAGS) -c $< log.o

log.o : log.c log.h
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f equations
	rm -f *.o
	rm -f *.s
	rm -f *.gcno
	rm -f *.gcda
	rm -f gmon.out
	rm -f *.out
	rm -f *.txt
	rm -rf .asan*
	rm -rf .ubsan*
	rm -rf .valgrind*
