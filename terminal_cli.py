"""Terminal interface and session routing; model/browser implementation stays separate."""

def run_terminal(controller) -> None:
    if controller.BACKEND not in {'ollama', 'llama', 'openai'}:
        raise ValueError(f'Unsupported DEEP_AGENT_BACKEND={controller.BACKEND!r}; use ollama or llama.')
    from kali_access import KaliAccess
    sessions = {'chat': [{'role': 'system', 'content': controller.CHAT_SYSTEM_PROMPT}], 'shell': [{'role': 'system', 'content': controller._shell_system_prompt()}]}
    mode = 'chat'
    controller.ACTIVE_MODE = mode
    controller.CONTEXT_LIMIT = None
    controller.CONTEXT_STATUS.update({'chat': None, 'shell': None})
    controller.REASONING_OVERRIDE.update({'effort': None, 'sticky': None})
    controller.EVIDENCE_LEDGER = controller.EvidenceLedger()
    controller.RESULT_ARCHIVE = controller.ResultArchive(controller.Path(controller.__file__).parent / 'runtime' / 'context' / f'{controller.uuid.uuid4().hex}.sqlite3')
    controller.CONTROLLER_DIAGNOSTICS = []
    kali = KaliAccess()
    previous_action = None
    previous_action_records: list[dict] = []
    previous_action_rejections: list[dict] = []
    previous_action_scope_target = None
    pending_scan_target = False
    pending_escalation_target = False
    print(f'{controller.ASSISTANT_NAME} chat mode. Type /shell for Kali commands, /help, or exit.')
    if controller._unrestricted_execution():
        print('Kali execution: unrestricted. Model commands run directly.', flush=True)
    while True:
        for history in sessions.values():
            try:
                controller.compact_tool_results(history, controller.EVIDENCE_LEDGER, controller.RESULT_ARCHIVE, controller.TOOL_RESULT_KEEP_LIMIT)
                history[:] = controller.bounded_history(history, *controller.history_limits(controller.CONTEXT_LIMIT), archive=controller.RESULT_ARCHIVE)
            except controller.ContextCapacityError:
                controller.compact_tool_results(history, controller.EVIDENCE_LEDGER, controller.RESULT_ARCHIVE, keep=0)
                try:
                    history[:] = controller.bounded_history(history, *controller.history_limits(controller.CONTEXT_LIMIT), archive=controller.RESULT_ARCHIVE)
                except controller.ContextCapacityError:
                    pass
        try:
            user = input(f'\n{controller._context_indicator(mode)}\nYou [{mode}]> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if controller.REASONING_ESCALATION.search(user) and controller.REASONING_OVERRIDE.get('sticky') != 'off':
            controller.REASONING_OVERRIDE['effort'] = 'high'
            print('[High reasoning requested for this turn.]', flush=True)
        else:
            controller.REASONING_OVERRIDE['effort'] = None
        if user.lower() in {'exit', 'quit', '/exit'}:
            break
        if user.lower() in {'/shell', '$shell'}:
            mode = 'shell'
            controller.ACTIVE_MODE = mode
            print('Kali shell mode. Enter a command, or ask for a lab action. Type /chat to return to chat.')
            if kali.sudo_mode:
                print('Sudo mode is active; Kali commands run through sudo until /user, /clear, or exit.')
            continue
        if user.lower() in {'/chat', '$chat'}:
            mode = 'chat'
            controller.ACTIVE_MODE = mode
            print('Chat mode. Type /shell to use Kali.')
            continue
        messages = sessions[mode]
        if mode == 'shell' and user.lower() in {'/user', '/unsudo'}:
            kali.clear_sudo_mode()
            controller._print_fallback(messages, 'Sudo mode disabled; subsequent Kali commands use the normal Kali account.')
            continue
        if user.lower() == '/clear':
            sessions[mode] = [{'role': 'system', 'content': controller.CHAT_SYSTEM_PROMPT if mode == 'chat' else controller._shell_system_prompt()}]
            controller.CONTEXT_STATUS[mode] = None
            if mode == 'shell':
                kali.clear_sudo_mode()
                previous_action = None
                previous_action_records = []
                previous_action_rejections = []
                previous_action_scope_target = None
                pending_scan_target = False
                pending_escalation_target = False
                controller.EVIDENCE_LEDGER.clear()
                controller.RESULT_ARCHIVE = controller.ResultArchive(controller.Path(controller.__file__).parent / 'runtime' / 'context' / f'{controller.uuid.uuid4().hex}.sqlite3')
                controller.CONTROLLER_DIAGNOSTICS.clear()
            print(f'{mode.capitalize()} history cleared.')
            continue
        if user.lower() == '/help':
            if mode == 'chat':
                print('Ask questions here; request a web search for current sources. Type /shell for Kali commands, SearchSploit, and lab actions. /context shows recent context usage. /evidence shows command evidence. /debug shows rejected or normalized tool calls and no-action replies. /clear resets this chat; exit quits.')
            elif controller._unrestricted_execution():
                print('Unrestricted Kali execution: enter commands or ask for an action. Scripts, pipelines, redirection, writes, and installs run directly. Type `sudo` for session sudo mode, `/user` to drop it, `/reasoning` for thinking settings, `/context` for context usage, `/evidence full` for command records, `/processes` for handles, `/chat` for chat, or exit. Ctrl+C interrupts the current request.')
            else:
                print('Enter a Kali command such as `ip a`, ask for web research or SearchSploit lookups, type `sudo` to enable session sudo mode, use `/user` to drop back, use `/kali COMMAND`, or use `/exploit PRIVATE_IPV4` for a scoped lab assessment. Long-running services use background handles; TTY apps use interactive handles. `/processes` lists controller-known handles. /context shows context usage; /evidence [full] shows command records; /debug shows rejected or normalized calls and no-action replies. Type /chat for ordinary chat. /clear resets this shell history; exit quits.')
            print('Use reasoning:low, reasoning:medium, or reasoning:high for thinking; reasoning:off disables it. /reasoning shows the setting.')
            print('Shell tools can retrieve earlier evidence with list_tool_results/read_tool_result. Persistent notes support keys, replacements and expiry; read_lab_notes shows IDs and status.')
            continue
        if user.lower() == '/context':
            print(controller._context_indicator(mode))
            continue
        reasoning_match = controller.re.fullmatch('(?:reasoning\\s*:\\s*|/reasoning\\s+)(off|low|medium|high)', user.strip(), controller.re.I)
        if reasoning_match:
            controller.REASONING_OVERRIDE['sticky'] = reasoning_match.group(1).lower()
            controller.REASONING_OVERRIDE['effort'] = None
            print(f"Reasoning set to {controller.REASONING_OVERRIDE['sticky']} for this session (chat and shell).", flush=True)
            continue
        if user.strip().lower() == '/reasoning':
            current = controller.REASONING_OVERRIDE.get('sticky')
            if current is None:
                if controller.BACKEND == 'ollama':
                    default = controller.os.environ.get('DEEP_AGENT_OLLAMA_THINK', '').strip().lower()
                elif controller.MODEL.lower().startswith('k2-horizon'):
                    default = controller._k2_reasoning_effort()
                elif controller.MODEL.lower().startswith('bonsai-2-27b'):
                    default = 'medium'
                else:
                    default = controller.os.environ.get('DEEP_AGENT_THINKING', '').strip().lower()
                if default in {'off', 'false', 'disabled', '0', 'no'}:
                    default = 'off'
                elif default in {'on', 'true', 'enabled', '1', 'yes'}:
                    default = 'on'
                current = f"{default or 'model default'} (agent default)"
            print(f'Reasoning setting: {current}. Use reasoning:low|medium|high|off.', flush=True)
            continue
        if controller.re.match('(?:reasoning\\s*:|/reasoning\\b)', user.strip(), controller.re.I):
            print('Use reasoning:low|medium|high|off (also /reasoning low|medium|high|off).', flush=True)
            continue
        if mode == 'shell' and user.lower() == '/processes':
            list_processes = getattr(kali, 'list_background_processes', None)
            result = list_processes() if callable(list_processes) else 'No process registry is available in this session.'
            controller._print_fallback(messages, result)
            continue
        if user.lower() == '/debug':
            if controller.CONTROLLER_DIAGNOSTICS:
                print(controller.json.dumps(controller.CONTROLLER_DIAGNOSTICS, ensure_ascii=False, indent=2))
            else:
                print('No rejected tool calls or no-action model responses have been recorded this session.')
            continue
        if user.lower() in {'/evidence', '/evidence full'}:
            print(controller.json.dumps(controller.EVIDENCE_LEDGER.snapshot(include_streams=user.lower().endswith(' full')), ensure_ascii=False, indent=2))
            continue
        if mode == 'chat':
            messages.append({'role': 'user', 'content': user})
            web_search_requested = bool(controller.WEB_SEARCH_INTENT.search(user))
            needs_kali = controller._needs_kali(user, None)
            if controller.EXPLOIT_REQUEST.fullmatch(user) or controller._direct_kali_command(user) or (needs_kali and (not controller.EXPLICIT_WEB_SEARCH.search(user))):
                controller._print_fallback(messages, 'Switch to /shell to run Kali commands and lab actions.')
                continue
            try:
                if web_search_requested:
                    controller._run_chat_web_search(messages)
                else:
                    reply = controller._model_chat(messages, tools=False, stream_output=True)
                    if reply.get('tool_calls'):
                        answer = 'This is chat mode. Type /shell to run a Kali command or lab action.'
                        print(f'\n{controller.ASSISTANT_NAME}> {answer}', flush=True)
                    else:
                        answer = controller.safe_answer(reply.get('content', ''))
                    messages.append({'role': 'assistant', 'content': answer})
            except KeyboardInterrupt:
                messages.pop()
                print('\nResponse stopped.')
            except Exception as exc:
                messages.pop()
                print(f'\nChat failed: {exc}')
            continue
        if controller._unrestricted_execution():
            messages.append({'role': 'user', 'content': user})
            connection_check = controller._local_kali_connection_request(user)
            direct_command = 'hostname && whoami' if connection_check else controller._direct_kali_command(user)
            try:
                controller._run_kali_turn(kali, messages, user, direct_command, connection_check=connection_check)
            except KeyboardInterrupt:
                print('\nResponse stopped.')
            except Exception as exc:
                print(f'\nAgent operation failed: {exc}')
            continue
        exploit_match = controller.EXPLOIT_REQUEST.fullmatch(user)
        if exploit_match:
            target_text = (exploit_match.group(1) or '').strip()
            target = controller._lab_exploit_target(target_text)
            if target is None and (not target_text or target_text.lower() in controller._EXPLOIT_PRONOUN_TARGETS):
                if previous_action_scope_target:
                    target = previous_action_scope_target
                else:
                    target = controller._recent_lab_target(previous_action_records)
            messages.append({'role': 'user', 'content': user})
            if target is None:
                controller._print_fallback(messages, 'Use `/exploit PRIVATE_IPV4` with one host in 10/8, 172.16/12, 192.168/16, or loopback. Public targets and hostnames are not accepted by this workflow.')
                continue
            exploit_request = f'Run the authorized lab vulnerability validation workflow against exactly {target}. Start with a bounded TCP service/version scan of this host only. Verify a candidate against observed product/version and local exploit research before testing it. If a relevant vulnerability is confirmed, attempt at most one low-impact proof of concept that does not open an interactive shell, change target files, persist, disable defenses, or access private data. Prefer a module check or non-destructive proof. Stop if validation would require a destructive action or a shell payload. Do not scan or connect to any other host. Report evidence and uncertainty.'
            previous_action = exploit_request
            previous_action_records = []
            previous_action_rejections = []
            previous_action_scope_target = target
            task_turn_start = len(messages)
            task_rejections: list[dict] = []
            controller._run_kali_turn(kali, messages, exploit_request, None, scope_target=target, rejection_history=task_rejections)
            previous_action_rejections = task_rejections
            previous_action_records = controller._merge_task_records(previous_action_records, controller._execution_records(messages[task_turn_start:]))
            continue
        if controller.re.fullmatch('(?:escalate\\s+(?:privil(?:e|a)ge|privileges?|sudo)|get\\s+root)', user, controller.re.I):
            messages.append({'role': 'user', 'content': user})
            if controller.re.search('\\bsudo\\b', user, controller.re.I):
                controller._print_fallback(messages, 'Type `sudo` to enable session sudo mode after a secure local password check, then enter Kali commands. The password stays in controller memory until `/user`, `/clear`, or exit. Type `/user` to drop back; interactive root shells are not supported.')
            else:
                pending_escalation_target = True
                controller._print_fallback(messages, 'Do you mean root on this Kali VM, or privilege escalation on a specified lab target?')
            continue
        if pending_escalation_target and user.lower() in {'kali', 'kali vm', 'this kali vm', 'local kali'}:
            pending_escalation_target = False
            messages.append({'role': 'user', 'content': user})
            controller._print_fallback(messages, 'Type `sudo` to enable session sudo mode after a secure local password check, then enter Kali commands. The password stays in controller memory until `/user`, `/clear`, or exit. Type `/user` to drop back; interactive root shells are not supported.')
            continue
        if pending_escalation_target and controller._is_acknowledgement_only(user):
            messages.append({'role': 'user', 'content': user})
            if user.lower() in {'no', 'nope'}:
                pending_escalation_target = False
                controller._print_fallback(messages, 'Privilege-escalation request canceled.')
            else:
                controller._print_fallback(messages, 'Please say `Kali VM` or provide a concrete authorized target host or IP; a yes/no reply does not identify a target.')
            continue
        selected = controller._selected_option(user, messages)
        continuation = controller._is_continuation(user)
        turn_scope_target = previous_action_scope_target if continuation or selected else None
        if not continuation and (not selected):
            previous_action_scope_target = None
        effective_request = selected or (previous_action if continuation and previous_action else user)
        escalation_target_reply = pending_escalation_target
        if pending_escalation_target:
            pending_escalation_target = False
            effective_request = f'Assess privilege escalation on the authorized lab target {user}'
        if controller.UNCLEAR_SCAN.fullmatch(user):
            messages.append({'role': 'user', 'content': user})
            answer = 'Which host should I scan? Give its IP address or hostname, or say ‘Kali itself’.'
            controller._print_fallback(messages, answer)
            pending_scan_target = True
            previous_action = user
            continue
        scan_target_reply = pending_scan_target
        if pending_scan_target:
            if user.lower() in {'cancel', 'never mind', 'nevermind'}:
                pending_scan_target = False
                messages.append({'role': 'user', 'content': user})
                controller._print_fallback(messages, 'Scan canceled.')
                continue
            if not controller._is_scan_target_reply(user):
                messages.append({'role': 'user', 'content': user})
                controller._print_fallback(messages, 'Please give the host IP address or hostname, or say ‘Kali itself’.')
                continue
            effective_request = f'scan ports {user}'
            pending_scan_target = False
        acknowledgement_as_conversation = controller._is_acknowledgement_only(user) and (not controller._assistant_requests_kali_action(messages))
        if controller.SHELL_CONVERSATION.fullmatch(user) or controller._is_quoted_phrase(user) or controller._is_conversational_question(user) or acknowledgement_as_conversation or (not controller.RESULT_QUESTION.fullmatch(user) and len(user.split()) == 1 and (not selected) and (not escalation_target_reply) and (not scan_target_reply) and (not controller._direct_kali_command(user)) and (not controller._needs_kali(user, previous_action))):
            messages.append({'role': 'user', 'content': user})
            try:
                reply = controller._model_chat(messages, tools=False, stream_output=False)
                if reply.get('tool_calls'):
                    controller._print_fallback(messages, 'No Kali command ran; this was handled as a conversational follow-up.')
                else:
                    answer = controller.safe_answer(reply.get('content', ''))
                    controller._print_fallback(messages, answer)
            except KeyboardInterrupt:
                messages.pop()
                print('\nResponse stopped.')
            except Exception as exc:
                messages.pop()
                print(f'\nChat failed: {exc}')
            continue
        continuing_task_records = previous_action_records if continuation or selected else []
        continuing_rejections = previous_action_rejections if continuation or selected else []
        if not continuation and (not selected) and (not controller.RESULT_QUESTION.fullmatch(user)):
            previous_action_records = []
            previous_action_rejections = []
        messages.append({'role': 'user', 'content': f'{user} (selected option: {selected})' if selected else user})
        install_intent = controller._package_install_request(effective_request)
        if install_intent:
            if continuation and continuing_task_records:
                controller._print_fallback(messages, 'No package command ran on this confirmation. The controller already ran the bounded install workflow for this goal on the previous turn and will not restart its package steps automatically. The previous result is above; make a new explicit request if you want to retry or use a different installation source.')
                continue
            previous_action = effective_request
            previous_action_rejections = []
            task_turn_start = len(messages)
            controller._run_package_install_workflow(kali, messages, install_intent)
            previous_action_records = controller._merge_task_records(continuing_task_records, controller._execution_records(messages[task_turn_start:]))
            continue
        requested_command = controller._direct_kali_command(user) or controller._default_scan_command(effective_request)
        connection_check = controller._local_kali_connection_request(user)
        if connection_check:
            requested_command = 'hostname && whoami'
        natural_action = controller._needs_kali(effective_request, previous_action)
        if not requested_command and (not selected) and (not natural_action):
            requested_command = controller._shell_command_fallback(user)
        try:
            if controller.RESULT_QUESTION.fullmatch(user):
                controller._summarize_results(messages, controller._execution_records(messages), user)
            else:
                previous_action = effective_request if not continuation else previous_action
                task_turn_start = len(messages)
                task_rejections = list(continuing_rejections)
                controller._run_kali_turn(kali, messages, effective_request, requested_command, scope_target=turn_scope_target, connection_check=connection_check, prior_task_records=continuing_task_records, prior_rejections=continuing_rejections, rejection_history=task_rejections)
                previous_action_rejections = task_rejections
                previous_action_records = controller._merge_task_records(continuing_task_records, controller._execution_records(messages[task_turn_start:]))
        except KeyboardInterrupt:
            print('\nResponse stopped.')
            if messages and messages[-1].get('role') == 'user':
                messages.pop()
        except Exception as exc:
            print(f'\nChat failed: {exc}')
            if messages and messages[-1].get('role') == 'user':
                messages.pop()
    kali.close()
