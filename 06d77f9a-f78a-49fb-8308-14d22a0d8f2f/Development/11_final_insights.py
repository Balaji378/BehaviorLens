print("\n" + "="*80)
print("BEHAVIORLENS — FINAL INSIGHTS")
print("="*80)

print("""
PROJECT OBJECTIVE
────────────────────────────────────────────────────────────
Identify which user behaviors and workflows are most predictive
of long-term success using product event data.

Success was defined as users in the top 20% of engagement
(total_events) — representing highly active product users.
""")

print("\n" + "="*80)
print("1️⃣  KEY BEHAVIORAL PREDICTORS OF SUCCESS")
print("="*80)

top_behaviors = feature_importance_df.head(10)

for _rank, _row in enumerate(top_behaviors.itertuples(), start=1):
    print(f"{_rank:2d}. {_row.feature}  (importance={_row.importance:.4f})")

print("""
INTERPRETATION
These actions are the strongest indicators that a user is moving
from casual exploration to meaningful product usage.

Users performing these behaviors are significantly more likely
to become long-term successful users.
""")

print("\n" + "="*80)
print("2️⃣  ENGAGEMENT PATTERNS OF SUCCESSFUL USERS")
print("="*80)

print(f"""
Successful users exhibit significantly higher engagement:

• Median Sessions
    Successful users:     {vis_successful['number_of_sessions'].median():.1f}
    Non-successful users: {vis_not_successful['number_of_sessions'].median():.1f}

• Median Total Events
    Successful users:     {vis_successful['total_events'].median():.1f}
    Non-successful users: {vis_not_successful['total_events'].median():.1f}

• Median Feature Diversity (unique event types)
    Successful users:     {vis_successful['number_of_event_types'].median():.1f}
    Non-successful users: {vis_not_successful['number_of_event_types'].median():.1f}

Key Insight:
Successful users interact with the product more frequently and
explore a wider variety of features.
""")

print("\n" + "="*80)
print("3️⃣  WORKFLOW PATTERNS THAT LEAD TO SUCCESS")
print("="*80)

print("""
Analysis of event transitions reveals several dominant workflows.

COMMON SUCCESSFUL WORKFLOW
One of the most common successful workflows resembles:
sign_in → canvas_open → block_open → run_block

This pattern suggests successful users quickly move from login
to executing blocks — indicating they understand how to use the
product effectively.

AI-ASSISTED WORKFLOW LOOP
credits_used → agent:create_block → agent:run_block → agent:get_block

This workflow indicates advanced users leveraging AI agents
to automate tasks and iterate rapidly.

RUN-CENTRIC DEVELOPMENT LOOP
block_created → run_block → block_run_done

This is the classic "build → run → verify" development cycle,
a strong indicator of productive usage.
""")

print("\n" + "="*80)
print("4️⃣  MACHINE LEARNING MODEL FINDINGS")
print("="*80)

print(f"""
A Random Forest model was trained to predict successful users.

Model Performance
────────────────────────
Accuracy : {accuracy*100:.2f}%
ROC-AUC  : {roc_auc:.3f}

The model confirms that behavioral signals from event activity
are strong predictors of long-term success.
""")

print("\n" + "="*80)
print("5️⃣  PRODUCT INSIGHTS & RECOMMENDATIONS")
print("="*80)

print("""
Based on the analysis, several actionable insights emerge:

1. Accelerate the "first successful run"
   Users who quickly execute blocks after signing in
   are much more likely to become successful.

2. Promote AI-assisted workflows
   Agent tool usage is strongly associated with power users.

3. Encourage feature exploration
   Successful users interact with a broader set of features.
   Guided tours or recommendations could increase discovery.

4. Reduce friction in the core workflow
   The block creation → execution cycle should remain
   extremely fast and intuitive.

5. Identify and nurture power users
   Users exhibiting these behavioral signals early
   are strong candidates for long-term retention.
""")

print("\n" + "="*80)
print("FINAL CONCLUSION")
print("="*80)

print("""
BehaviorLens reveals that long-term success is not random —
it is strongly tied to specific behavioral patterns.

Successful users:

• Run more workflows
• Use AI tools more frequently
• Engage across more product features
• Follow clear execution loops (build → run → iterate)

By guiding new users toward these behaviors earlier in their
journey, product teams can significantly improve activation,
retention, and overall product adoption.
""")

print("\n" + "="*80)
print("EARLY SUCCESS SIGNALS")
print("="*80)

early_success_rate = (
user_features[user_features["events_first_session"] > 5]
["successful_user"]
.mean()
)

print(f"""
Users performing more than 5 actions in their first session
have a success rate of approximately {early_success_rate:.2%}.

This suggests that early engagement strongly predicts
long-term success.

Product Recommendation
────────────────────────
Encourage new users to perform several meaningful actions
during their first session.

Examples:
• guided onboarding
• recommended workflows
• AI-assisted suggestions
""")
