#pragma once

#include "protocol.h"

void command_router_init(void);
void command_router_apply_action(vcp_action_id_t action_id, const char *keyword);
