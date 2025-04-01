#include "log.h"

void log_message(const char *message)
{
    printf("[%s] - %s", __TIME__, message);
    // fflush(stdout); // Ensure the message is printed immediately
}
