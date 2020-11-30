package com.itcast.actviti7demo.mapper;

import com.itcast.actviti7demo.model.HolidayAudit;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface HolidayAuditMapper {
    int insert(HolidayAudit record);

    int insertSelective(HolidayAudit record);
}