\restrict c2Cf8WMCrRmkDppJqqGfCj262sAgXlGLmyDThRkBSL9EDxj1lLyqpIbfvsWxFfo
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;
COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';
CREATE TABLE public.agent_events (
    id bigint NOT NULL,
    execution_id character varying(64) NOT NULL,
    seq integer NOT NULL,
    ts double precision NOT NULL,
    event character varying(64) NOT NULL,
    step_id character varying(32),
    message text NOT NULL,
    payload jsonb NOT NULL
);
CREATE SEQUENCE public.agent_events_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER SEQUENCE public.agent_events_id_seq OWNED BY public.agent_events.id;
CREATE TABLE public.agent_traces (
    id bigint NOT NULL,
    execution_id character varying(64) NOT NULL,
    ts double precision NOT NULL,
    agent character varying(64) NOT NULL,
    node character varying(64),
    step_id character varying(32),
    status character varying(32) NOT NULL,
    duration_ms double precision NOT NULL,
    confidence double precision,
    summary text NOT NULL,
    details jsonb NOT NULL
);
CREATE SEQUENCE public.agent_traces_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER SEQUENCE public.agent_traces_id_seq OWNED BY public.agent_traces.id;
CREATE TABLE public.application_environments (
    id character varying(64) NOT NULL,
    organization_id character varying(64) NOT NULL,
    application_id character varying(64) NOT NULL,
    name character varying(100) NOT NULL,
    base_url character varying(2048) NOT NULL,
    environment_type character varying(32) NOT NULL,
    allowed_domains jsonb NOT NULL,
    policy jsonb NOT NULL,
    variables jsonb NOT NULL,
    browser jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);
CREATE TABLE public.application_memory (
    id bigint NOT NULL,
    key character varying(128) NOT NULL,
    value jsonb NOT NULL,
    confidence double precision NOT NULL,
    updated_at timestamp with time zone NOT NULL,
    organization_id character varying(64) NOT NULL,
    project_id character varying(64) NOT NULL,
    application_id character varying(64) NOT NULL,
    environment_id character varying(64)
);
CREATE SEQUENCE public.application_memory_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER SEQUENCE public.application_memory_id_seq OWNED BY public.application_memory.id;
CREATE TABLE public.applications (
    id character varying(64) NOT NULL,
    organization_id character varying(64) NOT NULL,
    project_id character varying(64) NOT NULL,
    name character varying(200) NOT NULL,
    base_url character varying(2048) NOT NULL,
    host character varying(255) NOT NULL,
    known_routes jsonb NOT NULL,
    auth_behavior jsonb NOT NULL,
    fingerprint jsonb NOT NULL,
    updated_at timestamp with time zone NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);
CREATE TABLE public.approval_events (
    id character varying(64) NOT NULL,
    organization_id character varying(64) NOT NULL,
    execution_id character varying(64) NOT NULL,
    step_id character varying(32),
    kind character varying(32) NOT NULL,
    status character varying(16) NOT NULL,
    request jsonb NOT NULL,
    decided_by character varying(64),
    decision_note text,
    decided_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);
CREATE TABLE public.artifacts (
    id bigint NOT NULL,
    execution_id character varying(64) NOT NULL,
    step_id character varying(32),
    kind character varying(32) NOT NULL,
    name character varying(255) NOT NULL,
    uri character varying(512) NOT NULL,
    content_type character varying(128) NOT NULL,
    size bigint NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);
CREATE SEQUENCE public.artifacts_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER SEQUENCE public.artifacts_id_seq OWNED BY public.artifacts.id;
CREATE TABLE public.assertion_results (
    id bigint NOT NULL,
    execution_id character varying(64) NOT NULL,
    step_id character varying(32) NOT NULL,
    assertion_id character varying(64) NOT NULL,
    type character varying(48) NOT NULL,
    status character varying(16) NOT NULL,
    expected jsonb NOT NULL,
    actual jsonb NOT NULL,
    message text NOT NULL,
    evidence jsonb NOT NULL
);
CREATE SEQUENCE public.assertion_results_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER SEQUENCE public.assertion_results_id_seq OWNED BY public.assertion_results.id;
CREATE TABLE public.audit_logs (
    id bigint NOT NULL,
    organization_id character varying(64) NOT NULL,
    ts timestamp with time zone NOT NULL,
    actor character varying(128) NOT NULL,
    action character varying(64) NOT NULL,
    resource_type character varying(64) NOT NULL,
    resource_id character varying(64) NOT NULL,
    details jsonb NOT NULL
);
CREATE SEQUENCE public.audit_logs_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER SEQUENCE public.audit_logs_id_seq OWNED BY public.audit_logs.id;
CREATE TABLE public.browser_session_events (
    id bigint NOT NULL,
    execution_id character varying(64) NOT NULL,
    ts double precision NOT NULL,
    tool character varying(64) NOT NULL,
    ok boolean NOT NULL,
    latency_ms double precision NOT NULL,
    args jsonb NOT NULL,
    error text,
    failure_category character varying(64),
    page_url character varying(2048)
);
CREATE SEQUENCE public.browser_session_events_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER SEQUENCE public.browser_session_events_id_seq OWNED BY public.browser_session_events.id;
CREATE TABLE public.browser_sessions (
    id character varying(64) NOT NULL,
    execution_id character varying(64) NOT NULL,
    server_version character varying(64),
    options jsonb NOT NULL,
    started_at timestamp with time zone NOT NULL,
    ended_at timestamp with time zone
);
CREATE TABLE public.checkpoint_blobs (
    thread_id text NOT NULL,
    checkpoint_ns text DEFAULT ''::text NOT NULL,
    channel text NOT NULL,
    version text NOT NULL,
    type text NOT NULL,
    blob bytea
);
CREATE TABLE public.checkpoint_migrations (
    v integer NOT NULL
);
CREATE TABLE public.checkpoint_writes (
    thread_id text NOT NULL,
    checkpoint_ns text DEFAULT ''::text NOT NULL,
    checkpoint_id text NOT NULL,
    task_id text NOT NULL,
    idx integer NOT NULL,
    channel text NOT NULL,
    type text,
    blob bytea NOT NULL,
    task_path text DEFAULT ''::text NOT NULL
);
CREATE TABLE public.checkpoints (
    thread_id text NOT NULL,
    checkpoint_ns text DEFAULT ''::text NOT NULL,
    checkpoint_id text NOT NULL,
    parent_checkpoint_id text,
    type text,
    checkpoint jsonb NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb NOT NULL
);
CREATE TABLE public.environment_credentials (
    id character varying(64) NOT NULL,
    organization_id character varying(64) NOT NULL,
    project_id character varying(64) NOT NULL,
    environment_id character varying(64) NOT NULL,
    name character varying(128) NOT NULL,
    encrypted_value text NOT NULL,
    updated_at timestamp with time zone NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);
CREATE TABLE public.execution_metrics (
    id bigint NOT NULL,
    execution_id character varying(64) NOT NULL,
    name character varying(64) NOT NULL,
    value double precision NOT NULL,
    ts timestamp with time zone NOT NULL
);
CREATE SEQUENCE public.execution_metrics_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER SEQUENCE public.execution_metrics_id_seq OWNED BY public.execution_metrics.id;
CREATE TABLE public.execution_runs (
    id character varying(64) NOT NULL,
    organization_id character varying(64) NOT NULL,
    project_id character varying(64) NOT NULL,
    application_id character varying(64) NOT NULL,
    environment_id character varying(64),
    test_case_id character varying(64) NOT NULL,
    test_case_version integer NOT NULL,
    dataset_id character varying(64),
    suite_run_id character varying(64),
    retry_of character varying(64),
    status character varying(32) NOT NULL,
    outcome character varying(32),
    reason text NOT NULL,
    runner character varying(32),
    options jsonb NOT NULL,
    metrics jsonb NOT NULL,
    report_uri character varying(512),
    report_json_uri character varying(512),
    video_uri character varying(512),
    flaky boolean NOT NULL,
    created_by character varying(64),
    started_at timestamp with time zone,
    finished_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);
CREATE TABLE public.execution_schedules (
    id character varying(64) NOT NULL,
    organization_id character varying(64) NOT NULL,
    project_id character varying(64) NOT NULL,
    target_type character varying(16) NOT NULL,
    target_id character varying(64) NOT NULL,
    interval_minutes integer NOT NULL,
    enabled boolean NOT NULL,
    last_run_at timestamp with time zone,
    options jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);
CREATE TABLE public.execution_steps (
    id bigint NOT NULL,
    execution_id character varying(64) NOT NULL,
    step_id character varying(32) NOT NULL,
    index integer NOT NULL,
    original_text text NOT NULL,
    action character varying(32) NOT NULL,
    status character varying(32) NOT NULL,
    message text NOT NULL,
    attempts integer NOT NULL,
    recovered boolean NOT NULL,
    memory_used boolean NOT NULL,
    failure_category character varying(64),
    duration_ms double precision NOT NULL,
    result jsonb NOT NULL
);
CREATE SEQUENCE public.execution_steps_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER SEQUENCE public.execution_steps_id_seq OWNED BY public.execution_steps.id;
CREATE TABLE public.failure_memory (
    id bigint NOT NULL,
    execution_id character varying(64) NOT NULL,
    route_pattern character varying(512) NOT NULL,
    action_signature character varying(256) NOT NULL,
    failure_category character varying(64) NOT NULL,
    failure_signature character varying(64) NOT NULL,
    context jsonb NOT NULL,
    failed_strategies jsonb NOT NULL,
    successful_alternative jsonb,
    root_cause text NOT NULL,
    recovery_attempts integer NOT NULL,
    evidence jsonb NOT NULL,
    created_at timestamp with time zone NOT NULL,
    organization_id character varying(64) NOT NULL,
    project_id character varying(64) NOT NULL,
    application_id character varying(64) NOT NULL,
    environment_id character varying(64)
);
CREATE SEQUENCE public.failure_memory_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER SEQUENCE public.failure_memory_id_seq OWNED BY public.failure_memory.id;
CREATE TABLE public.interaction_patterns (
    id bigint NOT NULL,
    organization_id character varying(64) NOT NULL,
    application_id character varying(64) NOT NULL,
    pattern_type character varying(64) NOT NULL,
    signature character varying(256) NOT NULL,
    details jsonb NOT NULL,
    success_count integer NOT NULL,
    failure_count integer NOT NULL,
    confidence double precision NOT NULL,
    updated_at timestamp with time zone NOT NULL
);
CREATE SEQUENCE public.interaction_patterns_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER SEQUENCE public.interaction_patterns_id_seq OWNED BY public.interaction_patterns.id;
CREATE TABLE public.locator_memory (
    id character varying(64) NOT NULL,
    test_case_id character varying(64),
    route_pattern character varying(512) NOT NULL,
    page_fingerprint character varying(32) NOT NULL,
    action_signature character varying(256) NOT NULL,
    intent_text text NOT NULL,
    target_strategy jsonb NOT NULL,
    successful_execution_count integer NOT NULL,
    failed_execution_count integer NOT NULL,
    validated_run_count integer NOT NULL,
    confidence double precision NOT NULL,
    status character varying(16) NOT NULL,
    memory_version integer NOT NULL,
    supersedes character varying(64),
    embedding public.vector(256),
    last_execution_id character varying(64),
    last_verified_at timestamp with time zone NOT NULL,
    created_at timestamp with time zone NOT NULL,
    updated_at timestamp with time zone NOT NULL,
    organization_id character varying(64) NOT NULL,
    project_id character varying(64) NOT NULL,
    application_id character varying(64) NOT NULL,
    environment_id character varying(64)
);
CREATE TABLE public.memory_versions (
    id bigint NOT NULL,
    memory_id character varying(64) NOT NULL,
    memory_type character varying(32) NOT NULL,
    version integer NOT NULL,
    snapshot jsonb NOT NULL,
    change_reason text NOT NULL,
    execution_id character varying(64),
    created_at timestamp with time zone NOT NULL
);
CREATE SEQUENCE public.memory_versions_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER SEQUENCE public.memory_versions_id_seq OWNED BY public.memory_versions.id;
CREATE TABLE public.organizations (
    id character varying(64) NOT NULL,
    name character varying(200) NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);
CREATE TABLE public.page_fingerprints (
    id bigint NOT NULL,
    route_pattern character varying(512) NOT NULL,
    structure_hash character varying(32) NOT NULL,
    title_pattern character varying(512) NOT NULL,
    signature jsonb NOT NULL,
    seen_count integer NOT NULL,
    first_seen timestamp with time zone NOT NULL,
    last_seen timestamp with time zone NOT NULL,
    organization_id character varying(64) NOT NULL,
    project_id character varying(64) NOT NULL,
    application_id character varying(64) NOT NULL,
    environment_id character varying(64)
);
CREATE SEQUENCE public.page_fingerprints_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER SEQUENCE public.page_fingerprints_id_seq OWNED BY public.page_fingerprints.id;
CREATE TABLE public.page_memory (
    id bigint NOT NULL,
    route_pattern character varying(512) NOT NULL,
    structure_hash character varying(32) NOT NULL,
    page_kind character varying(32),
    title_pattern character varying(512) NOT NULL,
    forms jsonb NOT NULL,
    signature jsonb NOT NULL,
    embedding public.vector(256),
    seen_count integer NOT NULL,
    last_verified_at timestamp with time zone NOT NULL,
    organization_id character varying(64) NOT NULL,
    project_id character varying(64) NOT NULL,
    application_id character varying(64) NOT NULL,
    environment_id character varying(64)
);
CREATE SEQUENCE public.page_memory_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER SEQUENCE public.page_memory_id_seq OWNED BY public.page_memory.id;
CREATE TABLE public.projects (
    id character varying(64) NOT NULL,
    organization_id character varying(64) NOT NULL,
    name character varying(200) NOT NULL,
    description text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);
CREATE TABLE public.recovery_attempts (
    id bigint NOT NULL,
    execution_id character varying(64) NOT NULL,
    step_id character varying(32) NOT NULL,
    attempt integer NOT NULL,
    failure_category character varying(64) NOT NULL,
    strategy character varying(64) NOT NULL,
    outcome character varying(16) NOT NULL,
    rationale text NOT NULL,
    details jsonb NOT NULL
);
CREATE SEQUENCE public.recovery_attempts_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER SEQUENCE public.recovery_attempts_id_seq OWNED BY public.recovery_attempts.id;
CREATE TABLE public.test_case_versions (
    id character varying(64) NOT NULL,
    test_case_id character varying(64) NOT NULL,
    version integer NOT NULL,
    source_text text NOT NULL,
    source_format character varying(32) NOT NULL,
    spec jsonb NOT NULL,
    spec_hash character varying(64) NOT NULL,
    issues jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);
CREATE TABLE public.test_cases (
    id character varying(64) NOT NULL,
    organization_id character varying(64) NOT NULL,
    project_id character varying(64) NOT NULL,
    application_id character varying(64) NOT NULL,
    name character varying(300) NOT NULL,
    tags jsonb NOT NULL,
    current_version integer NOT NULL,
    archived boolean NOT NULL,
    updated_at timestamp with time zone NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);
CREATE TABLE public.test_data_sets (
    id character varying(64) NOT NULL,
    organization_id character varying(64) NOT NULL,
    test_case_id character varying(64) NOT NULL,
    name character varying(200) NOT NULL,
    variables jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);
CREATE TABLE public.test_execution_memory (
    id bigint NOT NULL,
    test_case_id character varying(64) NOT NULL,
    test_case_version integer NOT NULL,
    trajectory jsonb NOT NULL,
    success_count integer NOT NULL,
    failure_count integer NOT NULL,
    confidence double precision NOT NULL,
    last_execution_id character varying(64),
    last_verified_at timestamp with time zone NOT NULL,
    organization_id character varying(64) NOT NULL,
    project_id character varying(64) NOT NULL,
    application_id character varying(64) NOT NULL,
    environment_id character varying(64)
);
CREATE SEQUENCE public.test_execution_memory_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER SEQUENCE public.test_execution_memory_id_seq OWNED BY public.test_execution_memory.id;
CREATE TABLE public.test_steps (
    id bigint NOT NULL,
    test_case_version_id character varying(64) NOT NULL,
    step_id character varying(32) NOT NULL,
    index integer NOT NULL,
    action character varying(32) NOT NULL,
    original_text text NOT NULL,
    spec jsonb NOT NULL
);
CREATE SEQUENCE public.test_steps_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER SEQUENCE public.test_steps_id_seq OWNED BY public.test_steps.id;
CREATE TABLE public.test_suites (
    id character varying(64) NOT NULL,
    organization_id character varying(64) NOT NULL,
    project_id character varying(64) NOT NULL,
    name character varying(200) NOT NULL,
    description text NOT NULL,
    test_case_ids jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);
CREATE TABLE public.users (
    id character varying(64) NOT NULL,
    organization_id character varying(64) NOT NULL,
    email character varying(320) NOT NULL,
    name character varying(200) NOT NULL,
    role character varying(32) NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);
ALTER TABLE ONLY public.agent_events ALTER COLUMN id SET DEFAULT nextval('public.agent_events_id_seq'::regclass);
ALTER TABLE ONLY public.agent_traces ALTER COLUMN id SET DEFAULT nextval('public.agent_traces_id_seq'::regclass);
ALTER TABLE ONLY public.application_memory ALTER COLUMN id SET DEFAULT nextval('public.application_memory_id_seq'::regclass);
ALTER TABLE ONLY public.artifacts ALTER COLUMN id SET DEFAULT nextval('public.artifacts_id_seq'::regclass);
ALTER TABLE ONLY public.assertion_results ALTER COLUMN id SET DEFAULT nextval('public.assertion_results_id_seq'::regclass);
ALTER TABLE ONLY public.audit_logs ALTER COLUMN id SET DEFAULT nextval('public.audit_logs_id_seq'::regclass);
ALTER TABLE ONLY public.browser_session_events ALTER COLUMN id SET DEFAULT nextval('public.browser_session_events_id_seq'::regclass);
ALTER TABLE ONLY public.execution_metrics ALTER COLUMN id SET DEFAULT nextval('public.execution_metrics_id_seq'::regclass);
ALTER TABLE ONLY public.execution_steps ALTER COLUMN id SET DEFAULT nextval('public.execution_steps_id_seq'::regclass);
ALTER TABLE ONLY public.failure_memory ALTER COLUMN id SET DEFAULT nextval('public.failure_memory_id_seq'::regclass);
ALTER TABLE ONLY public.interaction_patterns ALTER COLUMN id SET DEFAULT nextval('public.interaction_patterns_id_seq'::regclass);
ALTER TABLE ONLY public.memory_versions ALTER COLUMN id SET DEFAULT nextval('public.memory_versions_id_seq'::regclass);
ALTER TABLE ONLY public.page_fingerprints ALTER COLUMN id SET DEFAULT nextval('public.page_fingerprints_id_seq'::regclass);
ALTER TABLE ONLY public.page_memory ALTER COLUMN id SET DEFAULT nextval('public.page_memory_id_seq'::regclass);
ALTER TABLE ONLY public.recovery_attempts ALTER COLUMN id SET DEFAULT nextval('public.recovery_attempts_id_seq'::regclass);
ALTER TABLE ONLY public.test_execution_memory ALTER COLUMN id SET DEFAULT nextval('public.test_execution_memory_id_seq'::regclass);
ALTER TABLE ONLY public.test_steps ALTER COLUMN id SET DEFAULT nextval('public.test_steps_id_seq'::regclass);
ALTER TABLE ONLY public.agent_events
    ADD CONSTRAINT agent_events_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.agent_traces
    ADD CONSTRAINT agent_traces_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.application_environments
    ADD CONSTRAINT application_environments_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.application_memory
    ADD CONSTRAINT application_memory_application_id_key_key UNIQUE (application_id, key);
ALTER TABLE ONLY public.application_memory
    ADD CONSTRAINT application_memory_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.applications
    ADD CONSTRAINT applications_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.approval_events
    ADD CONSTRAINT approval_events_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.artifacts
    ADD CONSTRAINT artifacts_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.assertion_results
    ADD CONSTRAINT assertion_results_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.audit_logs
    ADD CONSTRAINT audit_logs_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.browser_session_events
    ADD CONSTRAINT browser_session_events_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.browser_sessions
    ADD CONSTRAINT browser_sessions_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.checkpoint_blobs
    ADD CONSTRAINT checkpoint_blobs_pkey PRIMARY KEY (thread_id, checkpoint_ns, channel, version);
ALTER TABLE ONLY public.checkpoint_migrations
    ADD CONSTRAINT checkpoint_migrations_pkey PRIMARY KEY (v);
ALTER TABLE ONLY public.checkpoint_writes
    ADD CONSTRAINT checkpoint_writes_pkey PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx);
ALTER TABLE ONLY public.checkpoints
    ADD CONSTRAINT checkpoints_pkey PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id);
ALTER TABLE ONLY public.environment_credentials
    ADD CONSTRAINT environment_credentials_organization_id_project_id_environm_key UNIQUE (organization_id, project_id, environment_id, name);
ALTER TABLE ONLY public.environment_credentials
    ADD CONSTRAINT environment_credentials_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.execution_metrics
    ADD CONSTRAINT execution_metrics_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.execution_runs
    ADD CONSTRAINT execution_runs_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.execution_schedules
    ADD CONSTRAINT execution_schedules_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.execution_steps
    ADD CONSTRAINT execution_steps_execution_id_step_id_key UNIQUE (execution_id, step_id);
ALTER TABLE ONLY public.execution_steps
    ADD CONSTRAINT execution_steps_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.failure_memory
    ADD CONSTRAINT failure_memory_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.interaction_patterns
    ADD CONSTRAINT interaction_patterns_organization_id_application_id_pattern_key UNIQUE (organization_id, application_id, pattern_type, signature);
ALTER TABLE ONLY public.interaction_patterns
    ADD CONSTRAINT interaction_patterns_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.locator_memory
    ADD CONSTRAINT locator_memory_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.memory_versions
    ADD CONSTRAINT memory_versions_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.organizations
    ADD CONSTRAINT organizations_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.page_fingerprints
    ADD CONSTRAINT page_fingerprints_application_id_structure_hash_key UNIQUE (application_id, structure_hash);
ALTER TABLE ONLY public.page_fingerprints
    ADD CONSTRAINT page_fingerprints_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.page_memory
    ADD CONSTRAINT page_memory_application_id_route_pattern_structure_hash_key UNIQUE (application_id, route_pattern, structure_hash);
ALTER TABLE ONLY public.page_memory
    ADD CONSTRAINT page_memory_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.projects
    ADD CONSTRAINT projects_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.recovery_attempts
    ADD CONSTRAINT recovery_attempts_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.test_case_versions
    ADD CONSTRAINT test_case_versions_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.test_case_versions
    ADD CONSTRAINT test_case_versions_test_case_id_version_key UNIQUE (test_case_id, version);
ALTER TABLE ONLY public.test_cases
    ADD CONSTRAINT test_cases_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.test_data_sets
    ADD CONSTRAINT test_data_sets_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.test_execution_memory
    ADD CONSTRAINT test_execution_memory_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.test_execution_memory
    ADD CONSTRAINT test_execution_memory_test_case_id_test_case_version_key UNIQUE (test_case_id, test_case_version);
ALTER TABLE ONLY public.test_steps
    ADD CONSTRAINT test_steps_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.test_suites
    ADD CONSTRAINT test_suites_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);
CREATE INDEX checkpoint_blobs_thread_id_idx ON public.checkpoint_blobs USING btree (thread_id);
CREATE INDEX checkpoint_writes_thread_id_idx ON public.checkpoint_writes USING btree (thread_id);
CREATE INDEX checkpoints_thread_id_idx ON public.checkpoints USING btree (thread_id);
CREATE INDEX ix_agent_events_execution_id ON public.agent_events USING btree (execution_id);
CREATE INDEX ix_agent_traces_execution_id ON public.agent_traces USING btree (execution_id);
CREATE INDEX ix_application_environments_application_id ON public.application_environments USING btree (application_id);
CREATE INDEX ix_application_environments_organization_id ON public.application_environments USING btree (organization_id);
CREATE INDEX ix_application_memory_application_id ON public.application_memory USING btree (application_id);
CREATE INDEX ix_application_memory_organization_id ON public.application_memory USING btree (organization_id);
CREATE INDEX ix_application_memory_project_id ON public.application_memory USING btree (project_id);
CREATE INDEX ix_applications_host ON public.applications USING btree (host);
CREATE INDEX ix_applications_organization_id ON public.applications USING btree (organization_id);
CREATE INDEX ix_applications_project_id ON public.applications USING btree (project_id);
CREATE INDEX ix_approval_events_execution_id ON public.approval_events USING btree (execution_id);
CREATE INDEX ix_approval_events_organization_id ON public.approval_events USING btree (organization_id);
CREATE INDEX ix_approval_events_status ON public.approval_events USING btree (status);
CREATE INDEX ix_artifacts_execution_id ON public.artifacts USING btree (execution_id);
CREATE INDEX ix_artifacts_kind ON public.artifacts USING btree (kind);
CREATE INDEX ix_assertion_results_execution_id ON public.assertion_results USING btree (execution_id);
CREATE INDEX ix_audit_logs_organization_id ON public.audit_logs USING btree (organization_id);
CREATE INDEX ix_browser_session_events_execution_id ON public.browser_session_events USING btree (execution_id);
CREATE INDEX ix_browser_sessions_execution_id ON public.browser_sessions USING btree (execution_id);
CREATE INDEX ix_environment_credentials_organization_id ON public.environment_credentials USING btree (organization_id);
CREATE INDEX ix_execution_metrics_execution_id ON public.execution_metrics USING btree (execution_id);
CREATE INDEX ix_execution_metrics_name ON public.execution_metrics USING btree (name);
CREATE INDEX ix_execution_runs_application_id ON public.execution_runs USING btree (application_id);
CREATE INDEX ix_execution_runs_organization_id ON public.execution_runs USING btree (organization_id);
CREATE INDEX ix_execution_runs_outcome ON public.execution_runs USING btree (outcome);
CREATE INDEX ix_execution_runs_project_id ON public.execution_runs USING btree (project_id);
CREATE INDEX ix_execution_runs_status ON public.execution_runs USING btree (status);
CREATE INDEX ix_execution_runs_suite_run_id ON public.execution_runs USING btree (suite_run_id);
CREATE INDEX ix_execution_runs_test_case_id ON public.execution_runs USING btree (test_case_id);
CREATE INDEX ix_execution_schedules_organization_id ON public.execution_schedules USING btree (organization_id);
CREATE INDEX ix_execution_schedules_project_id ON public.execution_schedules USING btree (project_id);
CREATE INDEX ix_execution_steps_execution_id ON public.execution_steps USING btree (execution_id);
CREATE INDEX ix_failure_memory_action_signature ON public.failure_memory USING btree (action_signature);
CREATE INDEX ix_failure_memory_application_id ON public.failure_memory USING btree (application_id);
CREATE INDEX ix_failure_memory_execution_id ON public.failure_memory USING btree (execution_id);
CREATE INDEX ix_failure_memory_failure_category ON public.failure_memory USING btree (failure_category);
CREATE INDEX ix_failure_memory_failure_signature ON public.failure_memory USING btree (failure_signature);
CREATE INDEX ix_failure_memory_organization_id ON public.failure_memory USING btree (organization_id);
CREATE INDEX ix_failure_memory_project_id ON public.failure_memory USING btree (project_id);
CREATE INDEX ix_failure_memory_route_pattern ON public.failure_memory USING btree (route_pattern);
CREATE INDEX ix_interaction_patterns_application_id ON public.interaction_patterns USING btree (application_id);
CREATE INDEX ix_interaction_patterns_organization_id ON public.interaction_patterns USING btree (organization_id);
CREATE INDEX ix_locator_memory_action_signature ON public.locator_memory USING btree (action_signature);
CREATE INDEX ix_locator_memory_application_id ON public.locator_memory USING btree (application_id);
CREATE INDEX ix_locator_memory_confidence ON public.locator_memory USING btree (confidence);
CREATE INDEX ix_locator_memory_embedding ON public.locator_memory USING hnsw (embedding public.vector_cosine_ops);
CREATE INDEX ix_locator_memory_last_verified_at ON public.locator_memory USING btree (last_verified_at);
CREATE INDEX ix_locator_memory_lookup ON public.locator_memory USING btree (application_id, route_pattern, action_signature, status);
CREATE INDEX ix_locator_memory_memory_version ON public.locator_memory USING btree (memory_version);
CREATE INDEX ix_locator_memory_organization_id ON public.locator_memory USING btree (organization_id);
CREATE INDEX ix_locator_memory_page_fingerprint ON public.locator_memory USING btree (page_fingerprint);
CREATE INDEX ix_locator_memory_project_id ON public.locator_memory USING btree (project_id);
CREATE INDEX ix_locator_memory_route_pattern ON public.locator_memory USING btree (route_pattern);
CREATE INDEX ix_locator_memory_status ON public.locator_memory USING btree (status);
CREATE INDEX ix_locator_memory_test_case_id ON public.locator_memory USING btree (test_case_id);
CREATE INDEX ix_memory_versions_memory_id ON public.memory_versions USING btree (memory_id);
CREATE INDEX ix_page_fingerprints_application_id ON public.page_fingerprints USING btree (application_id);
CREATE INDEX ix_page_fingerprints_organization_id ON public.page_fingerprints USING btree (organization_id);
CREATE INDEX ix_page_fingerprints_project_id ON public.page_fingerprints USING btree (project_id);
CREATE INDEX ix_page_fingerprints_route_pattern ON public.page_fingerprints USING btree (route_pattern);
CREATE INDEX ix_page_fingerprints_structure_hash ON public.page_fingerprints USING btree (structure_hash);
CREATE INDEX ix_page_memory_application_id ON public.page_memory USING btree (application_id);
CREATE INDEX ix_page_memory_embedding ON public.page_memory USING hnsw (embedding public.vector_cosine_ops);
CREATE INDEX ix_page_memory_last_verified_at ON public.page_memory USING btree (last_verified_at);
CREATE INDEX ix_page_memory_organization_id ON public.page_memory USING btree (organization_id);
CREATE INDEX ix_page_memory_project_id ON public.page_memory USING btree (project_id);
CREATE INDEX ix_page_memory_route_pattern ON public.page_memory USING btree (route_pattern);
CREATE INDEX ix_page_memory_structure_hash ON public.page_memory USING btree (structure_hash);
CREATE INDEX ix_projects_organization_id ON public.projects USING btree (organization_id);
CREATE INDEX ix_recovery_attempts_execution_id ON public.recovery_attempts USING btree (execution_id);
CREATE INDEX ix_test_case_versions_spec_hash ON public.test_case_versions USING btree (spec_hash);
CREATE INDEX ix_test_case_versions_test_case_id ON public.test_case_versions USING btree (test_case_id);
CREATE INDEX ix_test_cases_application_id ON public.test_cases USING btree (application_id);
CREATE INDEX ix_test_cases_organization_id ON public.test_cases USING btree (organization_id);
CREATE INDEX ix_test_cases_project_id ON public.test_cases USING btree (project_id);
CREATE INDEX ix_test_data_sets_organization_id ON public.test_data_sets USING btree (organization_id);
CREATE INDEX ix_test_data_sets_test_case_id ON public.test_data_sets USING btree (test_case_id);
CREATE INDEX ix_test_execution_memory_application_id ON public.test_execution_memory USING btree (application_id);
CREATE INDEX ix_test_execution_memory_organization_id ON public.test_execution_memory USING btree (organization_id);
CREATE INDEX ix_test_execution_memory_project_id ON public.test_execution_memory USING btree (project_id);
CREATE INDEX ix_test_execution_memory_test_case_id ON public.test_execution_memory USING btree (test_case_id);
CREATE INDEX ix_test_steps_test_case_version_id ON public.test_steps USING btree (test_case_version_id);
CREATE INDEX ix_test_suites_organization_id ON public.test_suites USING btree (organization_id);
CREATE INDEX ix_test_suites_project_id ON public.test_suites USING btree (project_id);
CREATE INDEX ix_users_organization_id ON public.users USING btree (organization_id);
ALTER TABLE ONLY public.application_environments
    ADD CONSTRAINT application_environments_application_id_fkey FOREIGN KEY (application_id) REFERENCES public.applications(id);
ALTER TABLE ONLY public.application_environments
    ADD CONSTRAINT application_environments_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organizations(id);
ALTER TABLE ONLY public.applications
    ADD CONSTRAINT applications_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organizations(id);
ALTER TABLE ONLY public.applications
    ADD CONSTRAINT applications_project_id_fkey FOREIGN KEY (project_id) REFERENCES public.projects(id);
ALTER TABLE ONLY public.browser_sessions
    ADD CONSTRAINT browser_sessions_execution_id_fkey FOREIGN KEY (execution_id) REFERENCES public.execution_runs(id);
ALTER TABLE ONLY public.environment_credentials
    ADD CONSTRAINT environment_credentials_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organizations(id);
ALTER TABLE ONLY public.execution_runs
    ADD CONSTRAINT execution_runs_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organizations(id);
ALTER TABLE ONLY public.execution_schedules
    ADD CONSTRAINT execution_schedules_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organizations(id);
ALTER TABLE ONLY public.execution_steps
    ADD CONSTRAINT execution_steps_execution_id_fkey FOREIGN KEY (execution_id) REFERENCES public.execution_runs(id);
ALTER TABLE ONLY public.projects
    ADD CONSTRAINT projects_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organizations(id);
ALTER TABLE ONLY public.test_case_versions
    ADD CONSTRAINT test_case_versions_test_case_id_fkey FOREIGN KEY (test_case_id) REFERENCES public.test_cases(id);
ALTER TABLE ONLY public.test_cases
    ADD CONSTRAINT test_cases_application_id_fkey FOREIGN KEY (application_id) REFERENCES public.applications(id);
ALTER TABLE ONLY public.test_cases
    ADD CONSTRAINT test_cases_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organizations(id);
ALTER TABLE ONLY public.test_cases
    ADD CONSTRAINT test_cases_project_id_fkey FOREIGN KEY (project_id) REFERENCES public.projects(id);
ALTER TABLE ONLY public.test_data_sets
    ADD CONSTRAINT test_data_sets_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organizations(id);
ALTER TABLE ONLY public.test_data_sets
    ADD CONSTRAINT test_data_sets_test_case_id_fkey FOREIGN KEY (test_case_id) REFERENCES public.test_cases(id);
ALTER TABLE ONLY public.test_steps
    ADD CONSTRAINT test_steps_test_case_version_id_fkey FOREIGN KEY (test_case_version_id) REFERENCES public.test_case_versions(id);
ALTER TABLE ONLY public.test_suites
    ADD CONSTRAINT test_suites_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organizations(id);
ALTER TABLE ONLY public.test_suites
    ADD CONSTRAINT test_suites_project_id_fkey FOREIGN KEY (project_id) REFERENCES public.projects(id);
ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_organization_id_fkey FOREIGN KEY (organization_id) REFERENCES public.organizations(id);
\unrestrict c2Cf8WMCrRmkDppJqqGfCj262sAgXlGLmyDThRkBSL9EDxj1lLyqpIbfvsWxFfo
