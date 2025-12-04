# Copyright Sierra
"""
LLM-based Task Generator for Continual Learning Data Augmentation.

This module uses LLM to generate additional training tasks based on
existing task formats and domain policies.
"""

import json
import random
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from loguru import logger

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    logger.warning("tqdm not installed, progress bar will not be shown")

# Try to import litellm for LLM calls
try:
    import litellm
    from litellm import acompletion
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False
    logger.warning("litellm not installed, LLM generation will not work")


@dataclass
class TaskTemplate:
    """Template for generating tasks."""
    category: str
    description: str
    scenario_template: str
    expected_actions: List[str]
    nl_assertions: List[str]
    difficulty: str = "medium"  # easy, medium, hard


# Airline task templates
AIRLINE_TASK_TEMPLATES = [
    TaskTemplate(
        category="cancellation_refusal",
        description="User tries to cancel a flight that cannot be cancelled",
        scenario_template="""You want to cancel your reservation {reservation_id}.
{additional_context}
You don't want to cancel if you don't get a refund.""",
        expected_actions=["get_user_details", "get_reservation_details"],
        nl_assertions=["Agent should refuse to proceed with the cancellation."],
        difficulty="easy"
    ),
    TaskTemplate(
        category="delayed_flight_compensation",
        description="User complains about delayed flight and requests compensation",
        scenario_template="""You are contacting to complain about your delayed flight.
Your reservation is {reservation_id}.
{additional_context}
You want to get compensation for the delay.""",
        expected_actions=["get_user_details", "get_reservation_details", "get_flight_status"],
        nl_assertions=["Agent should verify the flight was delayed.", "Agent should determine appropriate compensation based on membership and class."],
        difficulty="medium"
    ),
    TaskTemplate(
        category="upgrade_cabin",
        description="User wants to upgrade their cabin class",
        scenario_template="""You want to upgrade your cabin class from {old_cabin} to {new_cabin} for reservation {reservation_id}.
{additional_context}""",
        expected_actions=["get_user_details", "get_reservation_details", "update_reservation_flights"],
        nl_assertions=["Agent should correctly calculate the price difference.", "Agent should update the reservation if user confirms."],
        difficulty="medium"
    ),
    TaskTemplate(
        category="add_baggage",
        description="User wants to add extra baggage to their reservation",
        scenario_template="""You want to add {num_bags} extra checked bags to your reservation {reservation_id}.
{additional_context}""",
        expected_actions=["get_user_details", "get_reservation_details", "update_reservation_baggages"],
        nl_assertions=["Agent should calculate baggage fees correctly.", "Agent should update baggage count if user confirms."],
        difficulty="easy"
    ),
    TaskTemplate(
        category="change_flight",
        description="User wants to change their flight to a different date or time",
        scenario_template="""You want to change your flight in reservation {reservation_id} to a different {change_type}.
{additional_context}""",
        expected_actions=["get_user_details", "get_reservation_details", "search_direct_flight", "update_reservation_flights"],
        nl_assertions=["Agent should search for available flights.", "Agent should update the reservation if user confirms."],
        difficulty="hard"
    ),
    TaskTemplate(
        category="passenger_update",
        description="User wants to update passenger information",
        scenario_template="""You need to update passenger information for reservation {reservation_id}.
{additional_context}""",
        expected_actions=["get_user_details", "get_reservation_details", "update_reservation_passengers"],
        nl_assertions=["Agent should not change the number of passengers.", "Agent should update passenger details if user confirms."],
        difficulty="easy"
    ),
    TaskTemplate(
        category="book_flight",
        description="User wants to book a new flight",
        scenario_template="""You want to book a {trip_type} flight from {origin} to {destination} on {date}.
You will have {num_passengers} passenger(s).
{additional_context}""",
        expected_actions=["get_user_details", "search_direct_flight", "book_reservation"],
        nl_assertions=["Agent should search for available flights.", "Agent should collect all required information.", "Agent should book if user confirms."],
        difficulty="hard"
    ),
    TaskTemplate(
        category="cancelled_flight_complaint",
        description="User complains about airline-cancelled flight",
        scenario_template="""Your flight in reservation {reservation_id} was cancelled by the airline.
{additional_context}
You want a refund or rebooking.""",
        expected_actions=["get_user_details", "get_reservation_details", "get_flight_status"],
        nl_assertions=["Agent should verify the flight was cancelled.", "Agent should offer appropriate options."],
        difficulty="medium"
    ),
    TaskTemplate(
        category="insurance_inquiry",
        description="User asks about insurance or tries to add/use insurance",
        scenario_template="""You have a question about travel insurance for reservation {reservation_id}.
{additional_context}""",
        expected_actions=["get_user_details", "get_reservation_details"],
        nl_assertions=["Agent should correctly explain insurance policy."],
        difficulty="easy"
    ),
    TaskTemplate(
        category="multi_topic",
        description="User has multiple requests in one conversation",
        scenario_template="""You have multiple things to discuss:
1. {topic1}
2. {topic2}
{additional_context}""",
        expected_actions=["get_user_details", "get_reservation_details"],
        nl_assertions=["Agent should handle all user requests.", "Agent should not miss any topic."],
        difficulty="hard"
    ),
]

# Retail task templates
RETAIL_TASK_TEMPLATES = [
    TaskTemplate(
        category="cancel_order",
        description="User wants to cancel a pending order",
        scenario_template="""You want to cancel your order {order_id}.
The reason is {reason}.
{additional_context}""",
        expected_actions=["find_user_id_by_name_zip", "get_order_details", "cancel_pending_order"],
        nl_assertions=["Agent should verify the order is pending.", "Agent should cancel if user confirms."],
        difficulty="easy"
    ),
    TaskTemplate(
        category="exchange_items",
        description="User wants to exchange items in a delivered order",
        scenario_template="""You received your order {order_id} and want to exchange {item_description} for {new_item_description}.
{additional_context}""",
        expected_actions=["find_user_id_by_name_zip", "get_order_details", "get_product_details", "exchange_delivered_order_items"],
        nl_assertions=["Agent should verify items can be exchanged.", "Agent should process exchange if user confirms."],
        difficulty="medium"
    ),
    TaskTemplate(
        category="return_items",
        description="User wants to return items from a delivered order",
        scenario_template="""You want to return some items from order {order_id}.
{additional_context}""",
        expected_actions=["find_user_id_by_name_zip", "get_order_details", "return_delivered_order_items"],
        nl_assertions=["Agent should verify order is delivered.", "Agent should process return if user confirms."],
        difficulty="easy"
    ),
    TaskTemplate(
        category="modify_address",
        description="User wants to modify shipping address of pending order",
        scenario_template="""You need to change the shipping address for order {order_id}.
New address: {new_address}
{additional_context}""",
        expected_actions=["find_user_id_by_name_zip", "get_order_details", "modify_pending_order_address"],
        nl_assertions=["Agent should verify order is pending.", "Agent should update address if user confirms."],
        difficulty="easy"
    ),
    TaskTemplate(
        category="modify_payment",
        description="User wants to change payment method for pending order",
        scenario_template="""You want to change the payment method for order {order_id}.
{additional_context}""",
        expected_actions=["find_user_id_by_name_zip", "get_order_details", "get_user_details", "modify_pending_order_payment"],
        nl_assertions=["Agent should verify order is pending.", "Agent should update payment if user confirms."],
        difficulty="medium"
    ),
    TaskTemplate(
        category="modify_items",
        description="User wants to modify items in a pending order",
        scenario_template="""You want to change some items in your pending order {order_id}.
{additional_context}""",
        expected_actions=["find_user_id_by_name_zip", "get_order_details", "get_product_details", "modify_pending_order_items"],
        nl_assertions=["Agent should verify order is pending.", "Agent should process modification if user confirms."],
        difficulty="medium"
    ),
    TaskTemplate(
        category="order_inquiry",
        description="User wants information about their order",
        scenario_template="""You have questions about order {order_id}.
{additional_context}""",
        expected_actions=["find_user_id_by_name_zip", "get_order_details"],
        nl_assertions=["Agent should provide accurate order information."],
        difficulty="easy"
    ),
    TaskTemplate(
        category="product_inquiry",
        description="User wants information about a product",
        scenario_template="""You want to know about {product_name}.
{additional_context}""",
        expected_actions=["find_user_id_by_name_zip", "list_all_product_types", "get_product_details"],
        nl_assertions=["Agent should provide accurate product information."],
        difficulty="easy"
    ),
    TaskTemplate(
        category="update_user_address",
        description="User wants to update their default address",
        scenario_template="""You want to update your default shipping address.
New address: {new_address}
{additional_context}""",
        expected_actions=["find_user_id_by_name_zip", "get_user_details", "modify_user_address"],
        nl_assertions=["Agent should update address if user confirms."],
        difficulty="easy"
    ),
    TaskTemplate(
        category="multi_item_exchange",
        description="User wants to exchange multiple items",
        scenario_template="""You received order {order_id} and want to exchange multiple items:
{exchange_details}
{additional_context}""",
        expected_actions=["find_user_id_by_name_zip", "get_order_details", "get_product_details", "exchange_delivered_order_items"],
        nl_assertions=["Agent should handle multiple item exchanges.", "Agent should process all exchanges in one call."],
        difficulty="hard"
    ),
]


class LLMTaskGenerator:
    """
    LLM-based task generator for creating synthetic training data.
    """

    def __init__(
        self,
        domain: str,
        db_path: Path,
        tasks_path: Path,
        policy_path: Path,
        model: str = "gpt-4o-mini",
        seed: int = 42,
    ):
        """
        Initialize the generator.

        Args:
            domain: Domain name (airline, retail)
            db_path: Path to database JSON
            tasks_path: Path to existing tasks JSON
            policy_path: Path to policy markdown
            model: LLM model to use
            seed: Random seed
        """
        self.domain = domain
        self.model = model
        self.seed = seed
        random.seed(seed)

        # Load data
        with open(db_path, 'r', encoding='utf-8') as f:
            self.db = json.load(f)

        with open(tasks_path, 'r', encoding='utf-8') as f:
            self.existing_tasks = json.load(f)

        with open(policy_path, 'r', encoding='utf-8') as f:
            self.policy = f.read()

        # Select templates
        if domain == "airline":
            self.templates = AIRLINE_TASK_TEMPLATES
        elif domain == "retail":
            self.templates = RETAIL_TASK_TEMPLATES
        else:
            raise ValueError(f"Unknown domain: {domain}")

        logger.info(f"Initialized LLMTaskGenerator for {domain}")
        logger.info(f"  Existing tasks: {len(self.existing_tasks)}")
        logger.info(f"  Templates: {len(self.templates)}")

    def _get_random_user(self) -> Dict[str, Any]:
        """Get a random user from the database."""
        users = self.db.get('users', {})
        if isinstance(users, dict):
            user_list = list(users.values())
        elif isinstance(users, list):
            user_list = users
        else:
            user_list = []
        if not user_list:
            return {}
        return random.choice(user_list)

    def _get_random_reservation(self) -> Optional[Dict[str, Any]]:
        """Get a random reservation from the database."""
        reservations = self.db.get('reservations', {})
        if isinstance(reservations, dict):
            res_list = list(reservations.values())
        elif isinstance(reservations, list):
            res_list = reservations
        else:
            res_list = []
        if not res_list:
            return None
        return random.choice(res_list)

    def _get_random_order(self) -> Optional[Dict[str, Any]]:
        """Get a random order from the database."""
        orders = self.db.get('orders', {})
        if isinstance(orders, dict):
            order_list = list(orders.values())
        elif isinstance(orders, list):
            order_list = orders
        else:
            order_list = []
        if not order_list:
            return None
        return random.choice(order_list)

    def _get_sample_tasks(self, n: int = 3) -> List[Dict[str, Any]]:
        """Get sample tasks for few-shot learning."""
        return random.sample(self.existing_tasks, min(n, len(self.existing_tasks)))

    def _create_generation_prompt(
        self,
        template: TaskTemplate,
        context: Dict[str, Any],
    ) -> str:
        """Create a prompt for LLM to generate a task."""
        sample_tasks = self._get_sample_tasks(3)

        prompt = f"""You are a task generator for a {self.domain} customer service benchmark.
Your goal is to generate realistic and diverse customer service scenarios.

## Domain Policy (Summary)
{self.policy[:3000]}...

## Task Template
Category: {template.category}
Description: {template.description}
Difficulty: {template.difficulty}

## Context Information
{json.dumps(context, indent=2)}

## Example Tasks (follow this format exactly)
{json.dumps(sample_tasks[:2], indent=2)}

## Instructions
Generate a NEW task following the exact JSON format of the examples above.
The task should be:
1. Realistic and diverse
2. Based on the template category: {template.category}
3. Using the provided context (user info, reservation/order info)
4. Different from the examples but following the same structure

Important:
- The task ID should be a unique string
- The user_scenario.instructions should have task_instructions, domain, reason_for_call, known_info, unknown_info
- The evaluation_criteria should have actions (list of expected tool calls), communicate_info (usually empty list), and nl_assertions (list of assertions about agent behavior)
- Make the scenario realistic and challenging

Generate ONLY valid JSON. Do not include any explanation or markdown formatting.
"""
        return prompt

    async def _generate_task_with_llm(
        self,
        template: TaskTemplate,
        task_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Generate a single task using LLM."""
        if not HAS_LITELLM:
            logger.error("litellm not available")
            return None

        # Build context
        context = {}
        if self.domain == "airline":
            user = self._get_random_user()
            reservation = self._get_random_reservation()

            # Handle payment_methods which could be dict or list
            pm = user.get("payment_methods", {})
            if isinstance(pm, dict):
                pm_keys = list(pm.keys())[:2]
            elif isinstance(pm, list):
                pm_keys = [p.get("id", str(p)) if isinstance(p, dict) else str(p) for p in pm[:2]]
            else:
                pm_keys = []

            context = {
                "user": {
                    "user_id": user.get("user_id"),
                    "name": user.get("name"),
                    "membership": user.get("membership", "regular"),
                    "payment_methods": pm_keys,
                },
                "reservation": reservation,
            }
        elif self.domain == "retail":
            user = self._get_random_user()
            order = self._get_random_order()

            # Handle payment_methods which could be dict or list
            pm = user.get("payment_methods", {})
            if isinstance(pm, dict):
                pm_keys = list(pm.keys())[:2]
            elif isinstance(pm, list):
                pm_keys = [p.get("id", str(p)) if isinstance(p, dict) else str(p) for p in pm[:2]]
            else:
                pm_keys = []

            context = {
                "user": {
                    "name": user.get("name"),
                    "address": user.get("address"),
                    "email": user.get("email"),
                    "payment_methods": pm_keys,
                },
                "order": order,
            }

        prompt = self._create_generation_prompt(template, context)

        try:
            response = await acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates JSON data."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                max_tokens=2000,
            )

            content = response.choices[0].message.content

            # Try to parse JSON
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            task = json.loads(content.strip())

            # Handle case where LLM returns a list instead of dict
            if isinstance(task, list):
                if len(task) == 0:
                    logger.warning(f"Generated task is empty list: {task_id}")
                    return None
                # Take the first element if it's a list
                task = task[0]

            if not isinstance(task, dict):
                logger.warning(f"Generated task is not a dict: {task_id}, type: {type(task)}")
                return None

            task["id"] = task_id

            # Validate basic structure
            if "user_scenario" not in task or "evaluation_criteria" not in task:
                logger.warning(f"Generated task missing required fields: {task_id}")
                return None

            # Ensure evaluation_criteria has required fields with correct types
            eval_criteria = task.get("evaluation_criteria", {})
            if not isinstance(eval_criteria.get("actions"), list):
                eval_criteria["actions"] = []
            if not isinstance(eval_criteria.get("communicate_info"), list):
                eval_criteria["communicate_info"] = []
            if eval_criteria.get("nl_assertions") is None:
                eval_criteria["nl_assertions"] = []
            elif not isinstance(eval_criteria.get("nl_assertions"), list):
                eval_criteria["nl_assertions"] = [eval_criteria["nl_assertions"]]
            task["evaluation_criteria"] = eval_criteria

            return task

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON for task {task_id}: {e}")
            logger.debug(f"Content was: {content[:500] if content else 'None'}...")
            return None
        except Exception as e:
            logger.error(f"Error generating task {task_id}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    async def generate_tasks(
        self,
        num_tasks: int,
        batch_size: int = 10,
        start_id: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple tasks.

        Args:
            num_tasks: Number of tasks to generate
            batch_size: Number of concurrent API calls
            start_id: Starting ID for new tasks

        Returns:
            List of generated tasks
        """
        if start_id is None:
            start_id = len(self.existing_tasks)

        generated_tasks = []

        # Create progress bar if tqdm is available
        if HAS_TQDM:
            pbar = tqdm(total=num_tasks, desc="Generating tasks", unit="task")

        for batch_start in range(0, num_tasks, batch_size):
            batch_end = min(batch_start + batch_size, num_tasks)
            batch_tasks = []

            for i in range(batch_start, batch_end):
                task_id = str(start_id + i)
                template = random.choice(self.templates)
                batch_tasks.append(self._generate_task_with_llm(template, task_id))

            # Run batch concurrently
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            batch_success_count = 0
            for result in results:
                if isinstance(result, dict) and result is not None:
                    generated_tasks.append(result)
                    batch_success_count += 1
                elif isinstance(result, Exception):
                    logger.error(f"Task generation failed: {result}")

            # Update progress bar
            if HAS_TQDM:
                pbar.update(len(results))
                pbar.set_postfix({"success": len(generated_tasks), "failed": batch_start + len(results) - len(generated_tasks)})
            else:
                logger.info(f"Generated {len(generated_tasks)}/{batch_start + len(results)} tasks (success/total)")

        if HAS_TQDM:
            pbar.close()

        logger.info(f"Final: Generated {len(generated_tasks)}/{num_tasks} tasks successfully")

        return generated_tasks

    def generate_tasks_sync(
        self,
        num_tasks: int,
        batch_size: int = 10,
        start_id: int = None,
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for generate_tasks."""
        import sys

        # Windows needs special event loop policy
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(self.generate_tasks(num_tasks, batch_size, start_id))
            return result
        finally:
            loop.close()

    def save_tasks(
        self,
        tasks: List[Dict[str, Any]],
        output_path: Path,
        append: bool = True,
    ) -> None:
        """
        Save generated tasks to file.

        Args:
            tasks: List of tasks to save
            output_path: Path to output file
            append: If True, append to existing tasks
        """
        try:
            logger.info(f"Attempting to save {len(tasks)} tasks to {output_path}")

            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if append and output_path.exists():
                logger.info(f"Appending to existing file: {output_path}")
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                all_tasks = existing + tasks
            else:
                logger.info(f"Creating new file: {output_path}")
                all_tasks = tasks

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_tasks, f, indent=4, ensure_ascii=False)

            logger.info(f"✓ Saved {len(tasks)} new tasks to {output_path}")
            logger.info(f"✓ Total tasks in file: {len(all_tasks)}")

        except Exception as e:
            logger.error(f"Failed to save tasks to {output_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise


def generate_airline_tasks(
    num_tasks: int = 307,
    output_path: Optional[Path] = None,
    model: str = "gpt-4o-mini",
) -> List[Dict[str, Any]]:
    """Generate airline tasks."""
    from tau2.utils.utils import DATA_DIR

    db_path = DATA_DIR / "tau2" / "domains" / "airline" / "db.json"
    tasks_path = DATA_DIR / "tau2" / "domains" / "airline" / "tasks.json"
    policy_path = DATA_DIR / "tau2" / "domains" / "airline" / "policy.md"

    if output_path is None:
        output_path = DATA_DIR / "tau2" / "domains" / "airline" / "tasks_augmented.json"

    logger.info(f"Generating airline tasks:")
    logger.info(f"  Database: {db_path}")
    logger.info(f"  Tasks: {tasks_path}")
    logger.info(f"  Policy: {policy_path}")
    logger.info(f"  Output: {output_path}")

    generator = LLMTaskGenerator(
        domain="airline",
        db_path=db_path,
        tasks_path=tasks_path,
        policy_path=policy_path,
        model=model,
    )

    tasks = generator.generate_tasks_sync(num_tasks)
    logger.info(f"Generated {len(tasks)} tasks, now saving...")

    generator.save_tasks(tasks, output_path, append=True)
    logger.info(f"Save complete!")

    return tasks


def generate_retail_tasks(
    num_tasks: int = 243,
    output_path: Optional[Path] = None,
    model: str = "gpt-4o-mini",
) -> List[Dict[str, Any]]:
    """Generate retail tasks."""
    from tau2.utils.utils import DATA_DIR

    db_path = DATA_DIR / "tau2" / "domains" / "retail" / "db.json"
    tasks_path = DATA_DIR / "tau2" / "domains" / "retail" / "tasks.json"
    policy_path = DATA_DIR / "tau2" / "domains" / "retail" / "policy.md"

    if output_path is None:
        output_path = DATA_DIR / "tau2" / "domains" / "retail" / "tasks_augmented.json"

    logger.info(f"Generating retail tasks:")
    logger.info(f"  Database: {db_path}")
    logger.info(f"  Tasks: {tasks_path}")
    logger.info(f"  Policy: {policy_path}")
    logger.info(f"  Output: {output_path}")

    generator = LLMTaskGenerator(
        domain="retail",
        db_path=db_path,
        tasks_path=tasks_path,
        policy_path=policy_path,
        model=model,
    )

    tasks = generator.generate_tasks_sync(num_tasks)
    logger.info(f"Generated {len(tasks)} tasks, now saving...")

    generator.save_tasks(tasks, output_path, append=True)
    logger.info(f"Save complete!")

    return tasks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic tasks using LLM")
    parser.add_argument("--domain", type=str, required=True, choices=["airline", "retail", "both"])
    parser.add_argument("--num-tasks", type=int, default=None, help="Number of tasks to generate")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--output", type=str, default=None, help="Output path")

    args = parser.parse_args()

    if args.domain == "airline" or args.domain == "both":
        num = args.num_tasks or 307
        output = Path(args.output) if args.output else None
        print(f"Generating {num} airline tasks...")
        generate_airline_tasks(num, output, args.model)

    if args.domain == "retail" or args.domain == "both":
        num = args.num_tasks or 243
        output = Path(args.output) if args.output else None
        print(f"Generating {num} retail tasks...")
        generate_retail_tasks(num, output, args.model)
