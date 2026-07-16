"""Config flow for Load Need Predictor.

The hub flow sets the global predict/capture times (and is reconfigurable). Each
load is added/edited as a ``ConfigSubentry`` via the per-load wizard, which shares
one ``init`` step between add and reconfigure (mirrors ha-load-scheduler).
"""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.config_entries import (
    SOURCE_USER,
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (
    CONF_CAPTURE_TIME,
    CONF_CONTROLLED_SWITCH_ENTITY,
    CONF_DEFICIT_CAP_MINUTES,
    CONF_DELIVERED_ENERGY_ENTITY,
    CONF_DELIVERED_RUNTIME_ENTITY,
    CONF_FIT_DAYS,
    CONF_FORECAST_DAYS,
    CONF_GUESTS_CALENDAR_ENTITY,
    CONF_HEATING_ACTIVE_ENTITY,
    CONF_MAX_MINUTES,
    CONF_MIN_MINUTES,
    CONF_NAME,
    CONF_OUTDOOR_TEMP_ENTITY,
    CONF_PERSON_ENTITIES,
    CONF_PREDICT_TIME,
    CONF_PRICE_ENTITY,
    CONF_RATED_POWER_KW,
    CONF_SUPPLY_TEMP_ENTITY,
    CONF_TANK_BOOST_SOC_PCT,
    CONF_TANK_COLD_IN_C,
    CONF_TANK_SETPOINT_C,
    CONF_TANK_VOLUME_L,
    CONF_TARGET_NUMBER_ENTITY,
    CONF_TEMP_HISTORY_ENTITY,
    CONF_WATER_TOTAL_ENTITY,
    CONF_WEATHER_ENTITY,
    CONF_WIND_ENTITY,
    DEFAULT_CAPTURE_TIME,
    DEFAULT_FIT_DAYS,
    DEFAULT_FORECAST_DAYS,
    DEFAULT_MAX_MINUTES,
    DEFAULT_MIN_MINUTES,
    DEFAULT_NAME,
    DEFAULT_PREDICT_TIME,
    DEFAULT_RATED_POWER_KW,
    DEFAULT_TANK_COLD_IN_C,
    DEFAULT_TANK_SETPOINT_C,
    DEFAULT_TANK_VOLUME_L,
    DOMAIN,
    SUBENTRY_TYPE_LOAD,
    SUBENTRY_TYPE_PRICE_FORECAST,
)

_SENSOR = selector.EntitySelector(selector.EntitySelectorConfig(domain="sensor"))
_TEMP_SENSOR = selector.EntitySelector(
    selector.EntitySelectorConfig(domain="sensor", device_class="temperature")
)


def _minutes_selector() -> selector.NumberSelector:
    return selector.NumberSelector(
        selector.NumberSelectorConfig(
            min=0, max=1440, step=5, unit_of_measurement="min", mode=selector.NumberSelectorMode.BOX
        )
    )


def _hub_schema(defaults: dict) -> vol.Schema:
    return vol.Schema(
        {
            vol.Optional(CONF_NAME, default=defaults.get(CONF_NAME, DEFAULT_NAME)): str,
            vol.Optional(
                CONF_PREDICT_TIME, default=defaults.get(CONF_PREDICT_TIME, DEFAULT_PREDICT_TIME)
            ): selector.TimeSelector(),
            vol.Optional(
                CONF_CAPTURE_TIME, default=defaults.get(CONF_CAPTURE_TIME, DEFAULT_CAPTURE_TIME)
            ): selector.TimeSelector(),
        }
    )


def _load_schema(defaults: dict) -> vol.Schema:
    def suggest(key: str) -> dict:
        return {"suggested_value": defaults.get(key)}

    return vol.Schema(
        {
            vol.Required(CONF_NAME, description=suggest(CONF_NAME)): str,
            vol.Optional(
                CONF_TARGET_NUMBER_ENTITY, description=suggest(CONF_TARGET_NUMBER_ENTITY)
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain="number")),
            vol.Required(
                CONF_DELIVERED_ENERGY_ENTITY, description=suggest(CONF_DELIVERED_ENERGY_ENTITY)
            ): _SENSOR,
            vol.Optional(
                CONF_DELIVERED_RUNTIME_ENTITY, description=suggest(CONF_DELIVERED_RUNTIME_ENTITY)
            ): _SENSOR,
            vol.Optional(
                CONF_CONTROLLED_SWITCH_ENTITY, description=suggest(CONF_CONTROLLED_SWITCH_ENTITY)
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain=["switch", "input_boolean"])
            ),
            vol.Optional(
                CONF_RATED_POWER_KW,
                default=defaults.get(CONF_RATED_POWER_KW, DEFAULT_RATED_POWER_KW),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0.1,
                    max=50,
                    step=0.1,
                    unit_of_measurement="kW",
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            vol.Optional(
                CONF_PERSON_ENTITIES, description=suggest(CONF_PERSON_ENTITIES)
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="person", multiple=True)
            ),
            vol.Optional(
                CONF_GUESTS_CALENDAR_ENTITY, description=suggest(CONF_GUESTS_CALENDAR_ENTITY)
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain="calendar")),
            vol.Optional(
                CONF_SUPPLY_TEMP_ENTITY, description=suggest(CONF_SUPPLY_TEMP_ENTITY)
            ): _TEMP_SENSOR,
            vol.Optional(
                CONF_OUTDOOR_TEMP_ENTITY, description=suggest(CONF_OUTDOOR_TEMP_ENTITY)
            ): _TEMP_SENSOR,
            vol.Optional(
                CONF_WATER_TOTAL_ENTITY, description=suggest(CONF_WATER_TOTAL_ENTITY)
            ): _SENSOR,
            vol.Optional(
                CONF_HEATING_ACTIVE_ENTITY, description=suggest(CONF_HEATING_ACTIVE_ENTITY)
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain="binary_sensor")),
            vol.Optional(
                CONF_TANK_VOLUME_L,
                default=defaults.get(CONF_TANK_VOLUME_L, DEFAULT_TANK_VOLUME_L),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=30,
                    max=1000,
                    step=10,
                    unit_of_measurement="L",
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            vol.Optional(
                CONF_TANK_SETPOINT_C,
                default=defaults.get(CONF_TANK_SETPOINT_C, DEFAULT_TANK_SETPOINT_C),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=40,
                    max=95,
                    step=1,
                    unit_of_measurement="°C",
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            vol.Optional(
                CONF_TANK_COLD_IN_C,
                default=defaults.get(CONF_TANK_COLD_IN_C, DEFAULT_TANK_COLD_IN_C),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=1,
                    max=25,
                    step=1,
                    unit_of_measurement="°C",
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            vol.Optional(
                CONF_TANK_BOOST_SOC_PCT, description=suggest(CONF_TANK_BOOST_SOC_PCT)
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0,
                    max=90,
                    step=5,
                    unit_of_measurement="%",
                    mode=selector.NumberSelectorMode.BOX,
                )
            ),
            vol.Optional(
                CONF_MIN_MINUTES, default=defaults.get(CONF_MIN_MINUTES, DEFAULT_MIN_MINUTES)
            ): _minutes_selector(),
            vol.Optional(
                CONF_MAX_MINUTES, default=defaults.get(CONF_MAX_MINUTES, DEFAULT_MAX_MINUTES)
            ): _minutes_selector(),
            vol.Optional(
                CONF_DEFICIT_CAP_MINUTES, description=suggest(CONF_DEFICIT_CAP_MINUTES)
            ): _minutes_selector(),
        }
    )


def _days_selector(maximum: int) -> selector.NumberSelector:
    return selector.NumberSelector(
        selector.NumberSelectorConfig(
            min=1,
            max=maximum,
            step=1,
            unit_of_measurement="d",
            mode=selector.NumberSelectorMode.BOX,
        )
    )


def _forecast_schema(defaults: dict) -> vol.Schema:
    def suggest(key: str) -> dict:
        return {"suggested_value": defaults.get(key)}

    return vol.Schema(
        {
            vol.Required(CONF_NAME, description=suggest(CONF_NAME)): str,
            vol.Required(CONF_PRICE_ENTITY, description=suggest(CONF_PRICE_ENTITY)): _SENSOR,
            vol.Required(CONF_WIND_ENTITY, description=suggest(CONF_WIND_ENTITY)): _SENSOR,
            vol.Required(
                CONF_WEATHER_ENTITY, description=suggest(CONF_WEATHER_ENTITY)
            ): selector.EntitySelector(selector.EntitySelectorConfig(domain="weather")),
            vol.Required(
                CONF_TEMP_HISTORY_ENTITY, description=suggest(CONF_TEMP_HISTORY_ENTITY)
            ): _TEMP_SENSOR,
            vol.Optional(
                CONF_FORECAST_DAYS, default=defaults.get(CONF_FORECAST_DAYS, DEFAULT_FORECAST_DAYS)
            ): _days_selector(7),
            vol.Optional(
                CONF_FIT_DAYS, default=defaults.get(CONF_FIT_DAYS, DEFAULT_FIT_DAYS)
            ): _days_selector(730),
        }
    )


def _clean(user_input: dict) -> dict:
    """Drop unset optionals so model defaults apply cleanly."""
    return {k: v for k, v in user_input.items() if v not in (None, "")}


class LoadNeedPredictorConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle the hub config flow."""

    VERSION = 1
    MINOR_VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Create the hub: set the predict/capture schedule."""
        if user_input is not None:
            return self.async_create_entry(
                title=user_input.get(CONF_NAME, DEFAULT_NAME), data=user_input
            )
        return self.async_show_form(step_id="user", data_schema=_hub_schema({}))

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Edit the hub's predict/capture schedule."""
        entry = self._get_reconfigure_entry()
        if user_input is not None:
            return self.async_update_reload_and_abort(entry, data_updates=user_input)
        defaults = {**entry.data, **(user_input or {})}
        return self.async_show_form(step_id="reconfigure", data_schema=_hub_schema(defaults))

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Loads and the price forecast are both added as subentries of the hub."""
        return {
            SUBENTRY_TYPE_LOAD: LoadSubentryFlowHandler,
            SUBENTRY_TYPE_PRICE_FORECAST: PriceForecastSubentryFlowHandler,
        }


class LoadSubentryFlowHandler(ConfigSubentryFlow):
    """Add or reconfigure a load (one shared ``init`` step)."""

    _defaults: dict

    @property
    def _is_new(self) -> bool:
        return self.source == SOURCE_USER

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        self._defaults = {}
        return await self.async_step_init(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        self._defaults = dict(self._get_reconfigure_subentry().data)
        return await self.async_step_init(user_input)

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        if user_input is not None:
            data = _clean(user_input)
            if self._is_new:
                return self.async_create_entry(title=data[CONF_NAME], data=data)
            return self.async_update_and_abort(
                self._get_entry(),
                self._get_reconfigure_subentry(),
                title=data[CONF_NAME],
                data=data,
            )
        return self.async_show_form(step_id="init", data_schema=_load_schema(self._defaults))


class PriceForecastSubentryFlowHandler(ConfigSubentryFlow):
    """Add or reconfigure the beyond-horizon price forecast (one ``init`` step)."""

    _defaults: dict

    @property
    def _is_new(self) -> bool:
        return self.source == SOURCE_USER

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        self._defaults = {}
        return await self.async_step_init(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        self._defaults = dict(self._get_reconfigure_subentry().data)
        return await self.async_step_init(user_input)

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        if user_input is not None:
            data = _clean(user_input)
            if self._is_new:
                return self.async_create_entry(title=data[CONF_NAME], data=data)
            return self.async_update_and_abort(
                self._get_entry(),
                self._get_reconfigure_subentry(),
                title=data[CONF_NAME],
                data=data,
            )
        return self.async_show_form(step_id="init", data_schema=_forecast_schema(self._defaults))
